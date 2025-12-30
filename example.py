from nanovllm import LLM, SamplingParams
# 1. 引入 SpeculativeConfig
from nanovllm.engine.speculative import SpeculativeConfig

def main():
    # 假设你有两个模型路径
    target_model_path = "/mnt/raid5/wrz_data/huggingface/Qwen3-8B"  # 大模型
    draft_model_path = "/mnt/raid5/wrz_data/huggingface/Qwen3-0.6B"   # 小模型 (草稿)

    # 2. 定义推测配置
    spec_config = SpeculativeConfig(
        max_draft_tokens=4,          # 每次生成 4 个草稿
        draft_model=draft_model_path # 指定草稿模型路径
    )

    # 3. 初始化 LLM 时传入 speculative_config
    llm = LLM(
        model=target_model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        speculative_config=spec_config  # <--- 关键开启开关
    )

    # 4. 采样参数 (目前你的策略只支持 Greedy，所以 temperature 建议设为 0 或配合你的 verify 逻辑)
    # 注意：你的 strategy 代码目前是用 argmax 验证的，这意味着它执行的是 Greedy Verification。
    # 如果 temperature > 0，实际上是在做 Lookahead Decoding (如果草稿碰巧命中了采样结果)。
    # 建议先用 temp=0 测试正确性。
    sampling_params = SamplingParams(temperature=1e-5, max_tokens=100)
    
    prompts = ["The future of AI is"]
    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0]["text"])

if __name__ == "__main__":
    main()
