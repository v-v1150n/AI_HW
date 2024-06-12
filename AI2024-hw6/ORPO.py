import gc
import os
import json
import utils
import torch
import wandb
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel
from trl import ORPOConfig, ORPOTrainer
from unsloth import is_bfloat16_supported
# max_seq_length = 4096 # Supports RoPE Scaling interally, so choose any!



def ORPO_train(args, output_dir):
    wandb.login(key=args.wandb_token)
    wandb.init(project="hw6_rlhf",
               name=f"{args.exp_name}_{args.model_name.split('/')[1]}")

    torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device\n")

    # Model
    # model, tokenizer = FastLanguageModel.from_pretrained(args.model_name,...)
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = 'unsloth/mistral-7b-v0.3-bnb-4bit',
        # model_name = 'unsloth/llama-3-8b-bnb-4bit',
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = 4096, #0609-1
        dtype = None,
        # dtype = torch.float16, #0609-2
        load_in_4bit = True,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    #max_seq_length =2048 過長的序列會增加計算成本，但過短的序列可能會丟失重要的上下文信息。
    #dtype = torch.float16 使用浮點16位或 BF16 來提高計算效率並減少內存使用。

    # Load dataset
    # ================================DO NOT CHANGE!================================
    dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    dataset = dataset.map(utils.format_prompt, fn_kwargs={"EOS_TOKEN": EOS_TOKEN})
    dataset = dataset.train_test_split(test_size=0.01)

    with open("./test_prompt.json", 'r') as f:
        test_data = json.load(f)
    # ================================DO NOT CHANGE!================================

    # Perform model patching and add fast LoRA weights
    # model = FastLanguageModel.get_peft_model(model,...)
    model = FastLanguageModel.get_peft_model(
        model,
        # r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        r = 64, #0609-1
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        # lora_alpha = 16,
        lora_alpha = 64, #0609-1
        # lora_dropout = 0, # Supports any, but = 0 is optimized
        lora_dropout = 0.1, #0609-1
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    #r = 64 調高可以提升模型的靈活性，讓模型能夠捕捉到更細微的特徵，但也會使訓練和推理變得更慢。
    #lora_alpha=64 較高的 lora_alpha 會增加 LoRA 層的影響，從而可能提升性能，但也會使模型更加敏感於過擬合。
    #lora_dropout = 0.1 如果出現過擬合問題，可以考慮增大 lora_dropout 的值，例如設為 0.1 或 0.2。


    # Training arguments
    training_args = ORPOConfig(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_prompt_length,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        output_dir=output_dir,
        save_strategy=args.save_strategy,
        report_to=args.report_to
    )

    # Trainer
    orpo_trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=training_args,
    )

    # Fine-tune model with DPO
    orpo_trainer.train()

    # Save model
    model.save_pretrained(output_dir)

    # Inference
    FastLanguageModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer)

    output_data = []

    for data in tqdm(test_data):
        print("=============Generated Answer After Fine-tuning=============\n")
        print(f"Question {data['id']}:\n"+data["prompt"])
        prompt = utils.alpaca_prompt.format(
            "You are a helpful assistant chatbot.",  # Instruction
            data["prompt"],  # Input
            "",  # Response, leave empty for generation
        )
        prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_sequences = model.generate(**prompt, streamer=text_streamer,
                           max_new_tokens=500)
        # Decode the generated output
        generated_text = tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True)[0]
        print("==============================================================\n")

        # Store the output in a list
        output_data.append({
            "id": data["id"],
            "prompt": data["prompt"],
            "generated_text": generated_text
        })
    
    # Ensure the submission directory exists
    submission_dir = "submission"
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    # Write the output data to a JSON file
    output_file = os.path.join(submission_dir, f"ORPO_{args.model_name.split('/')[1]}.json")
    utils.write_json(output_data, output_file)
    # Flush memory
    del orpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()
