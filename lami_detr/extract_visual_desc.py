import os
import json
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI


# =======================
# DeepSeek Client
# =======================
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# =======================
# Prompt
# =======================
def system_prompt_cn(category_names):
    return [
        {
            "role": "system",
            "content": (
                "You are a vision expert. "
                "You must ONLY output valid JSON. "
                "Do not include explanations or markdown."
            )
        },
        {
            "role": "user",
            "content": f"""
Task:
Convert the given category names into visual feature descriptions.

Input format:
One or multiple category names separated by '/'.
Example: lemur/television

Output format (STRICT JSON ONLY):
{{
  "category1": ["feature1", "feature2", ...],
  "category2": ["feature1", "feature2", ...]
}}

Rules:
- Use English phrases only
- Visual attributes only
- Each feature as an independent phrase
- No analysis, no markdown, no extra text

Input:
{category_names}
"""
        }
    ]


# =======================
# Dataset
# =======================
class CustomDataset(Dataset):
    def __init__(self, image_dir, tag_dir):
        self.image_dir = image_dir
        self.tag_dir = tag_dir
        self.data = os.listdir(image_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        json_name = img_name.split('.')[0] + '.json'
        tag_path = os.path.join(self.tag_dir, json_name)

        general_tag = json.load(open(tag_path))

        return {
            "img_name": img_name,
            "json_name": json_name,
            "img_path": os.path.join(self.image_dir, img_name),
            "general_tag": general_tag
        }

    def collate_fn(self, batch):
        return {
            key: [sample[key] for sample in batch]
            for key in ["img_name", "json_name", "img_path", "general_tag"]
        }


# =======================
# Call DeepSeek
# =======================
def call_deepseek(messages, max_retries=3, sleep_time=1.0):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.2,   # 降低随机性，提升 JSON 稳定性
                top_p=0.9,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Retry {attempt+1}] DeepSeek API error: {e}")
            time.sleep(sleep_time)
    return None


# =======================
# Main
# =======================
def main():
    image_dir = "./output/images"
    tag_dir = "./output/tags"
    OUT_PATH = "./output/visual_descs"

    os.makedirs(OUT_PATH, exist_ok=True)

    dataset = CustomDataset(image_dir, tag_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    start_time = time.time()

    for batch in tqdm(dataloader):
        for i in range(len(batch["json_name"])):
            save_path = os.path.join(OUT_PATH, batch["json_name"][i])

            # 跳过已处理
            if os.path.exists(save_path):
                continue

            general_tag = batch["general_tag"][i]

            messages = system_prompt_cn(general_tag)
            result = call_deepseek(messages)

            if result is None:
                print(f"[Failed] {batch['json_name'][i]}")
                continue

            # 清洗 & 解析 JSON
            try:
                result = result.strip()
                if result.startswith("```"):
                    result = result.replace("```json", "").replace("```", "").strip()

                parsed = json.loads(result)

                with open(save_path, "w") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"[Parse Error] {batch['json_name'][i]}: {e}")
                print("Raw output:", result)

            # API 限速保护（很重要）
            time.sleep(0.3)

    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
