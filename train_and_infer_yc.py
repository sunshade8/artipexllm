import sys
import os
import json
import torch
import transformers

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType


# ✅ 1️⃣ Hugging Face 로그인 (토큰을 환경 변수에서 불러오기)
def login_huggingface():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Token: ")
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    from huggingface_hub import login
    login(token=token)


# ✅ 2️⃣ 데이터셋 클래스 정의
class InstructDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        instruction = record.get("instruction", "").strip()
        input_text = record.get("input", "").strip()
        output_text = record.get("output", "").strip()
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: "
        full_text = prompt + output_text

        tokenized_full = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = tokenized_full["input_ids"].squeeze(0)
        attention_mask = tokenized_full["attention_mask"].squeeze(0)

        tokenized_prompt = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        prompt_length = tokenized_prompt["input_ids"].squeeze(0).shape[0]

        labels = input_ids.clone()
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }





# ✅ 4️⃣ 모델 학습
def train_model(model_name, dataset_file, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    
        # ✅ 3️⃣ 데이터 Collator 정의
    def custom_collate_fn(batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_masks = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks_padded = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "labels": labels_padded,
        }

    # LoRA 설정
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 옵티마이저 설정 (IPEX 최적화 없이)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, weight_decay=1e-3)

    # 데이터셋 로드
    train_dataset = InstructDataset(dataset_file, tokenizer, max_length=512)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_steps=1000,
        save_total_limit=2,
    )

    # Trainer 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)  # optimizer만 명시
    )
    trainer.train()


# ✅ 5️⃣ 모델 추론 (Fine-tuned & Original 모델 비교)
def infer(model_path, input_text):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_length=100, do_sample=True, temperature=0.7,
        top_k=50, top_p=0.9, repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ✅ 6️⃣ 실행 메인 함수
def main():
    login_huggingface()

    model_name = "EleutherAI/gpt-neo-1.3B"
    dataset_file = "chat/validation_data_1.jsonl"
    fine_tuned_model_path = "./fine_tuned_gpt-neo-1.3B/03.23"

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("🚀 Training the model...")
        train_model(model_name, dataset_file, fine_tuned_model_path)

    elif len(sys.argv) > 1 and sys.argv[1] == "infer":
        input_text = input("Enter your question: ")
        print("🧠 Fine-tuned Model Response:", infer(fine_tuned_model_path, input_text))

        print("🧠 Original Model Response:", infer(model_name, input_text))

    else:
        print("Usage: python train_and_infer.py [train|infer]")
        print("  train  → Train the model")
        print("  infer  → Run inference")


if __name__ == "__main__":
    main()





