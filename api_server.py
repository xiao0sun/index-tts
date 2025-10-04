"""
IndexTTS2 FastAPI服务器
提供REST API接口用于文本转语音生成
"""
import os
import sys
import time
import base64
import argparse
from typing import Optional, List
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from indextts.infer_v2 import IndexTTS2

# 配置类（用于测试环境）
class Config:
    def __init__(self):
        self.port = int(os.getenv("API_PORT", "8000"))
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.model_dir = os.getenv("MODEL_DIR", "./checkpoints")
        self.fp16 = os.getenv("API_FP16", "false").lower() == "true"
        self.deepspeed = os.getenv("API_DEEPSPEED", "false").lower() == "true"
        self.cuda_kernel = os.getenv("API_CUDA_KERNEL", "false").lower() == "true"
        self.output_dir = os.getenv("API_OUTPUT_DIR", "./outputs/api")

# 全局配置实例
cmd_args = Config()

# 创建输出目录
os.makedirs(cmd_args.output_dir, exist_ok=True)
os.makedirs("./temp/uploads", exist_ok=True)

# 初始化FastAPI应用
app = FastAPI(
    title="IndexTTS2 API",
    description="IndexTTS2情感表达和持续时间可控的零样本文本转语音API",
    version="2.0.0"
)

# 全局TTS模型实例
tts_model: Optional[IndexTTS2] = None


# Pydantic模型定义
class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    spk_audio_base64: Optional[str] = Field(None, description="说话人音频提示(base64编码)")
    emo_audio_base64: Optional[str] = Field(None, description="情感音频提示(base64编码)")
    emo_weight: float = Field(0.65, description="情感权重", ge=0.0, le=1.0)
    emo_vector: Optional[List[float]] = Field(None, description="8维情感向量[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]")
    use_emo_text: bool = Field(False, description="从文本推断情感")
    emo_text: Optional[str] = Field(None, description="情感描述文本")
    use_random: bool = Field(False, description="情感随机采样")
    do_sample: bool = Field(True, description="是否进行采样")
    temperature: float = Field(0.8, description="采样温度", ge=0.1, le=2.0)
    top_p: float = Field(0.8, description="Top-p采样参数", ge=0.0, le=1.0)
    top_k: int = Field(30, description="Top-k采样参数", ge=0, le=100)
    num_beams: int = Field(3, description="束搜索数量", ge=1, le=10)
    repetition_penalty: float = Field(10.0, description="重复惩罚", ge=0.1, le=20.0)
    length_penalty: float = Field(0.0, description="长度惩罚", ge=-2.0, le=2.0)
    max_mel_tokens: int = Field(1500, description="最大MEL token数", ge=50, le=3000)
    max_text_tokens_per_segment: int = Field(120, description="每段最大文本token数", ge=20, le=300)
    return_base64: bool = Field(False, description="是否返回base64编码的音频")


class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None
    duration: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    device: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化TTS模型"""
    global tts_model

    # 检查模型目录
    if not os.path.exists(cmd_args.model_dir):
        raise RuntimeError(f"模型目录 {cmd_args.model_dir} 不存在，请先下载模型")

    # 检查必需文件
    required_files = ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]
    for file in required_files:
        file_path = os.path.join(cmd_args.model_dir, file)
        if not os.path.exists(file_path):
            raise RuntimeError(f"必需文件 {file_path} 不存在，请下载完整模型")

    print("正在加载IndexTTS2模型...")
    tts_model = IndexTTS2(
        model_dir=cmd_args.model_dir,
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
        use_fp16=cmd_args.fp16,
        use_deepspeed=cmd_args.deepspeed,
        use_cuda_kernel=cmd_args.cuda_kernel,
    )
    print(f"模型加载完成！设备: {tts_model.device}, 版本: {tts_model.model_version or '1.0'}")


@app.get("/", response_model=HealthResponse)
async def root():
    """根路径健康检查"""
    return {
        "status": "running",
        "model_loaded": tts_model is not None,
        "model_version": str(tts_model.model_version) if tts_model and tts_model.model_version else None,
        "device": str(tts_model.device) if tts_model else None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if tts_model is not None else "model not loaded",
        "model_loaded": tts_model is not None,
        "model_version": str(tts_model.model_version) if tts_model and tts_model.model_version else None,
        "device": str(tts_model.device) if tts_model else None
    }


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    文本转语音API

    支持多种情感控制方式：
    1. 默认情感（与说话人音频相同）
    2. 情感音频提示
    3. 情感向量控制（8维）
    4. 文本情感推断
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS模型未加载")

    start_time = time.time()

    try:
        # 处理说话人音频（必需）
        spk_audio_path = None
        if request.spk_audio_base64:
            spk_audio_path = f"./temp/uploads/spk_{int(time.time() * 1000)}.wav"
            audio_data = base64.b64decode(request.spk_audio_base64)
            with open(spk_audio_path, "wb") as f:
                f.write(audio_data)

        # 处理情感音频（可选）
        emo_audio_path = None
        if request.emo_audio_base64:
            emo_audio_path = f"./temp/uploads/emo_{int(time.time() * 1000)}.wav"
            audio_data = base64.b64decode(request.emo_audio_base64)
            with open(emo_audio_path, "wb") as f:
                f.write(audio_data)

        # 生成输出路径
        output_path = os.path.join(cmd_args.output_dir, f"tts_{int(time.time() * 1000)}.wav")

        # 准备推理参数
        kwargs = {
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "top_k": request.top_k if request.top_k > 0 else None,
            "temperature": request.temperature,
            "length_penalty": request.length_penalty,
            "num_beams": request.num_beams,
            "repetition_penalty": request.repetition_penalty,
            "max_mel_tokens": request.max_mel_tokens,
        }

        # 处理情感向量
        emo_vec = None
        if request.emo_vector and len(request.emo_vector) == 8:
            emo_vec = tts_model.normalize_emo_vec(request.emo_vector, apply_bias=True)

        # 调用TTS推理
        output = tts_model.infer(
            spk_audio_prompt=spk_audio_path,
            text=request.text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_path,
            emo_alpha=request.emo_weight,
            emo_vector=emo_vec,
            use_emo_text=request.use_emo_text,
            emo_text=request.emo_text if request.emo_text else None,
            use_random=request.use_random,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
            **kwargs
        )

        # 清理临时文件
        if spk_audio_path and os.path.exists(spk_audio_path):
            os.remove(spk_audio_path)
        if emo_audio_path and os.path.exists(emo_audio_path):
            os.remove(emo_audio_path)

        duration = time.time() - start_time

        # 返回结果
        response = {
            "success": True,
            "message": "语音生成成功",
            "duration": duration
        }

        if request.return_base64:
            # 返回base64编码的音频
            with open(output, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode()
            response["audio_base64"] = audio_base64
        else:
            # 返回文件路径
            response["audio_path"] = output

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/tts/file")
async def text_to_speech_file(
    text: str = Form(..., description="要合成的文本"),
    spk_audio: UploadFile = File(..., description="说话人音频提示文件"),
    emo_audio: Optional[UploadFile] = File(None, description="情感音频提示文件"),
    emo_weight: float = Form(0.65, description="情感权重"),
    temperature: float = Form(0.8, description="采样温度"),
    max_mel_tokens: int = Form(1500, description="最大MEL token数")
):
    """
    文本转语音API（文件上传方式）

    直接上传音频文件而不是base64编码
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS模型未加载")

    start_time = time.time()

    try:
        # 保存上传的说话人音频
        spk_audio_path = f"./temp/uploads/spk_{int(time.time() * 1000)}.wav"
        with open(spk_audio_path, "wb") as f:
            f.write(await spk_audio.read())

        # 保存情感音频（如果提供）
        emo_audio_path = None
        if emo_audio:
            emo_audio_path = f"./temp/uploads/emo_{int(time.time() * 1000)}.wav"
            with open(emo_audio_path, "wb") as f:
                f.write(await emo_audio.read())

        # 生成输出路径
        output_path = os.path.join(cmd_args.output_dir, f"tts_{int(time.time() * 1000)}.wav")

        # 调用TTS推理
        output = tts_model.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_path,
            emo_alpha=emo_weight,
            temperature=temperature,
            max_mel_tokens=max_mel_tokens
        )

        # 清理临时文件
        if os.path.exists(spk_audio_path):
            os.remove(spk_audio_path)
        if emo_audio_path and os.path.exists(emo_audio_path):
            os.remove(emo_audio_path)

        # 返回生成的音频文件
        return FileResponse(
            output,
            media_type="audio/wav",
            filename=f"generated_{int(time.time())}.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取生成的音频文件"""
    file_path = os.path.join(cmd_args.output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(file_path, media_type="audio/wav")


if __name__ == "__main__":
    # 命令行参数解析（仅在直接运行时）
    parser = argparse.ArgumentParser(
        description="IndexTTS2 API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API服务器主机地址")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="模型检查点目录")
    parser.add_argument("--fp16", action="store_true", default=False, help="使用FP16推理")
    parser.add_argument("--deepspeed", action="store_true", default=False, help="使用DeepSpeed加速")
    parser.add_argument("--cuda_kernel", action="store_true", default=False, help="使用CUDA内核")
    parser.add_argument("--output_dir", type=str, default="./outputs/api", help="API生成音频输出目录")
    args = parser.parse_args()

    # 更新全局配置
    cmd_args.port = args.port
    cmd_args.host = args.host
    cmd_args.model_dir = args.model_dir
    cmd_args.fp16 = args.fp16
    cmd_args.deepspeed = args.deepspeed
    cmd_args.cuda_kernel = args.cuda_kernel
    cmd_args.output_dir = args.output_dir

    # 重新创建输出目录
    os.makedirs(cmd_args.output_dir, exist_ok=True)

    print(f"启动IndexTTS2 API服务器于 {cmd_args.host}:{cmd_args.port}")
    print(f"模型目录: {cmd_args.model_dir}")
    print(f"输出目录: {cmd_args.output_dir}")
    print(f"FP16: {cmd_args.fp16}, DeepSpeed: {cmd_args.deepspeed}, CUDA Kernel: {cmd_args.cuda_kernel}")

    uvicorn.run(
        app,
        host=cmd_args.host,
        port=cmd_args.port,
        log_level="info"
    )
