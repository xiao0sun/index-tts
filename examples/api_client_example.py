#!/usr/bin/env python3
"""
IndexTTS2 API客户端示例

这个脚本演示了如何使用IndexTTS2 API进行文本转语音生成。
支持多种调用方式和情感控制方法。

运行方式:
    python examples/api_client_example.py
"""
import requests
import base64
import json
import os
import time
from pathlib import Path


class IndexTTSAPIClient:
    """IndexTTS2 API客户端"""

    def __init__(self, base_url="http://localhost:8000"):
        """
        初始化API客户端

        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self):
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"健康检查失败: {e}")
            return None

    def tts_json(
        self,
        text,
        spk_audio_path=None,
        spk_audio_base64=None,
        emo_audio_path=None,
        emo_audio_base64=None,
        emo_vector=None,
        emo_weight=0.65,
        use_emo_text=False,
        emo_text=None,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        max_mel_tokens=1500,
        return_base64=False,
        **kwargs
    ):
        """
        使用JSON方式调用TTS API

        Args:
            text: 要合成的文本
            spk_audio_path: 说话人音频文件路径
            spk_audio_base64: 说话人音频base64编码（与spk_audio_path二选一）
            emo_audio_path: 情感音频文件路径
            emo_audio_base64: 情感音频base64编码
            emo_vector: 8维情感向量
            emo_weight: 情感权重
            use_emo_text: 是否使用文本情感推断
            emo_text: 情感描述文本
            temperature: 采样温度
            top_p: Top-p采样
            top_k: Top-k采样
            max_mel_tokens: 最大MEL token数
            return_base64: 是否返回base64编码的音频
            **kwargs: 其他参数

        Returns:
            API响应字典
        """
        # 处理说话人音频
        if spk_audio_path and not spk_audio_base64:
            with open(spk_audio_path, "rb") as f:
                spk_audio_base64 = base64.b64encode(f.read()).decode()

        # 处理情感音频
        if emo_audio_path and not emo_audio_base64:
            with open(emo_audio_path, "rb") as f:
                emo_audio_base64 = base64.b64encode(f.read()).decode()

        # 构建请求数据
        request_data = {
            "text": text,
            "spk_audio_base64": spk_audio_base64,
            "emo_weight": emo_weight,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_mel_tokens": max_mel_tokens,
            "return_base64": return_base64,
            **kwargs
        }

        if emo_audio_base64:
            request_data["emo_audio_base64"] = emo_audio_base64

        if emo_vector:
            request_data["emo_vector"] = emo_vector

        if use_emo_text:
            request_data["use_emo_text"] = True
            if emo_text:
                request_data["emo_text"] = emo_text

        # 发送请求
        try:
            response = requests.post(
                f"{self.base_url}/tts",
                json=request_data,
                timeout=120  # 2分钟超时
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return {"success": False, "message": str(e)}

    def tts_file(
        self,
        text,
        spk_audio_path,
        emo_audio_path=None,
        emo_weight=0.65,
        temperature=0.8,
        max_mel_tokens=1500,
        output_path="output.wav"
    ):
        """
        使用文件上传方式调用TTS API

        Args:
            text: 要合成的文本
            spk_audio_path: 说话人音频文件路径
            emo_audio_path: 情感音频文件路径（可选）
            emo_weight: 情感权重
            temperature: 采样温度
            max_mel_tokens: 最大MEL token数
            output_path: 输出文件路径

        Returns:
            成功返回True，失败返回False
        """
        try:
            files = {
                "spk_audio": open(spk_audio_path, "rb")
            }

            if emo_audio_path:
                files["emo_audio"] = open(emo_audio_path, "rb")

            data = {
                "text": text,
                "emo_weight": str(emo_weight),
                "temperature": str(temperature),
                "max_mel_tokens": str(max_mel_tokens)
            }

            response = requests.post(
                f"{self.base_url}/tts/file",
                files=files,
                data=data,
                timeout=120
            )
            response.raise_for_status()

            # 保存返回的音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"音频已保存到: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            return False
        finally:
            # 关闭文件
            for f in files.values():
                if hasattr(f, 'close'):
                    f.close()

    def get_audio(self, filename, output_path):
        """
        获取已生成的音频文件

        Args:
            filename: 音频文件名
            output_path: 保存路径

        Returns:
            成功返回True，失败返回False
        """
        try:
            response = requests.get(f"{self.base_url}/audio/{filename}", timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"音频已下载到: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}")
            return False


def example_basic_tts():
    """示例1: 基本的TTS生成"""
    print("\n=== 示例1: 基本TTS生成 ===")

    client = IndexTTSAPIClient()

    # 健康检查
    health = client.health_check()
    print(f"服务状态: {health}")

    if not health or not health.get("model_loaded"):
        print("警告: 模型未加载")
        return

    # 生成语音
    result = client.tts_json(
        text="你好，这是一个基本的测试",
        spk_audio_path="examples/voice_01.wav",
        return_base64=False
    )

    if result["success"]:
        print(f"✓ 生成成功: {result['audio_path']}")
        print(f"  耗时: {result['duration']:.2f}秒")
    else:
        print(f"✗ 生成失败: {result['message']}")


def example_emotion_audio():
    """示例2: 使用情感音频控制"""
    print("\n=== 示例2: 使用情感音频控制 ===")

    client = IndexTTSAPIClient()

    result = client.tts_json(
        text="今天天气真好啊！",
        spk_audio_path="examples/voice_01.wav",
        emo_audio_path="examples/voice_01.wav",  # 使用相同音频作为情感参考
        emo_weight=0.7,
        return_base64=False
    )

    if result["success"]:
        print(f"✓ 生成成功: {result['audio_path']}")
        print(f"  情感权重: 0.7")
    else:
        print(f"✗ 生成失败: {result['message']}")


def example_emotion_vector():
    """示例3: 使用情感向量控制"""
    print("\n=== 示例3: 使用情感向量控制 ===")

    client = IndexTTSAPIClient()

    # 8维情感向量: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    emotions = {
        "快乐": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5],
        "悲伤": [0.0, 0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.2],
        "惊讶": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.3],
        "平静": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }

    for emotion_name, emotion_vec in emotions.items():
        print(f"\n生成{emotion_name}情感...")

        result = client.tts_json(
            text=f"这是一个测试{emotion_name}情感的例子",
            spk_audio_path="examples/voice_01.wav",
            emo_vector=emotion_vec,
            emo_weight=0.7,
            return_base64=False
        )

        if result["success"]:
            print(f"✓ {emotion_name}生成成功: {result['audio_path']}")
        else:
            print(f"✗ {emotion_name}生成失败")


def example_file_upload():
    """示例4: 文件上传方式"""
    print("\n=== 示例4: 文件上传方式 ===")

    client = IndexTTSAPIClient()

    success = client.tts_file(
        text="使用文件上传方式生成语音",
        spk_audio_path="examples/voice_01.wav",
        emo_weight=0.65,
        output_path="outputs/file_upload_example.wav"
    )

    if success:
        print("✓ 文件上传方式生成成功")
    else:
        print("✗ 文件上传方式生成失败")


def example_custom_parameters():
    """示例5: 自定义生成参数"""
    print("\n=== 示例5: 自定义生成参数 ===")

    client = IndexTTSAPIClient()

    # 更高的temperature会产生更多样化的结果
    result = client.tts_json(
        text="测试不同的生成参数效果",
        spk_audio_path="examples/voice_01.wav",
        temperature=1.2,
        top_p=0.9,
        top_k=50,
        num_beams=5,
        repetition_penalty=8.0,
        max_mel_tokens=1200,
        return_base64=False
    )

    if result["success"]:
        print(f"✓ 自定义参数生成成功")
        print(f"  参数: temperature=1.2, top_p=0.9, top_k=50")
    else:
        print(f"✗ 生成失败")


def example_long_text():
    """示例6: 长文本生成"""
    print("\n=== 示例6: 长文本生成 ===")

    client = IndexTTSAPIClient()

    long_text = """
    IndexTTS2是一个开创性的情感表达和持续时间可控的自回归零样本文本转语音系统。
    它支持语音克隆、情感控制和精确持续时间控制，同时实现了情感表达与说话人身份的解耦。
    这个系统可以生成高质量的语音，并且支持中英文混合建模。
    """

    result = client.tts_json(
        text=long_text.strip(),
        spk_audio_path="examples/voice_01.wav",
        max_text_tokens_per_segment=100,
        return_base64=False
    )

    if result["success"]:
        print(f"✓ 长文本生成成功")
        print(f"  耗时: {result['duration']:.2f}秒")
    else:
        print(f"✗ 生成失败")


def example_base64_response():
    """示例7: 获取base64编码的音频"""
    print("\n=== 示例7: Base64编码响应 ===")

    client = IndexTTSAPIClient()

    result = client.tts_json(
        text="测试base64返回格式",
        spk_audio_path="examples/voice_01.wav",
        return_base64=True
    )

    if result["success"]:
        print(f"✓ Base64生成成功")
        print(f"  Base64长度: {len(result['audio_base64'])} 字符")

        # 解码并保存
        audio_data = base64.b64decode(result["audio_base64"])
        output_path = "outputs/base64_example.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_data)
        print(f"  已解码并保存到: {output_path}")
    else:
        print(f"✗ 生成失败")


def example_batch_generation():
    """示例8: 批量生成"""
    print("\n=== 示例8: 批量生成 ===")

    client = IndexTTSAPIClient()

    texts = [
        "第一段测试文本",
        "第二段测试文本",
        "第三段测试文本",
    ]

    print(f"批量生成 {len(texts)} 段语音...")
    start_time = time.time()

    for i, text in enumerate(texts, 1):
        result = client.tts_json(
            text=text,
            spk_audio_path="examples/voice_01.wav",
            return_base64=False
        )

        if result["success"]:
            print(f"  [{i}/{len(texts)}] ✓ 生成成功")
        else:
            print(f"  [{i}/{len(texts)}] ✗ 生成失败")

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}秒")
    print(f"平均每段: {total_time/len(texts):.2f}秒")


def main():
    """运行所有示例"""
    print("IndexTTS2 API客户端示例")
    print("=" * 50)

    # 检查示例音频文件
    if not os.path.exists("examples/voice_01.wav"):
        print("错误: 找不到示例音频文件 examples/voice_01.wav")
        print("请确保在项目根目录运行此脚本")
        return

    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)

    # 运行示例
    try:
        example_basic_tts()
        example_emotion_audio()
        example_emotion_vector()
        example_file_upload()
        example_custom_parameters()
        example_long_text()
        example_base64_response()
        example_batch_generation()

        print("\n" + "=" * 50)
        print("所有示例执行完成！")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
