import logging
from pathlib import Path

import soundfile

from src.inference import infer_tool
from src.inference.infer_tool import Svc
from src.quality_presets import BEST_QUALITY_PRESET

logging.getLogger('numba').setLevel(logging.WARNING)
CHUNKS_TEMP_PATH = Path(__file__).resolve().parents[1] / "inference" / "chunks_temp.json"
chunks_dict = infer_tool.read_temp(CHUNKS_TEMP_PATH)



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # 一定要设置的部分
    parser.add_argument('-m', '--model_path', type=str, default="model_assets/workspaces/44k/G_37600.pth", help='模型路径')
    parser.add_argument('-c', '--config_path', type=str, default="model_assets/workspaces/44k/config.json", help='配置文件路径')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='wav文件名列表，放在 inference_data/inputs 文件夹下')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    parser.add_argument('-s', '--speaker', type=str, default='buyizi', help='当前模型音色名称')
    
    # 极致音质配置：固定参数
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='[极致音质配置] 转换歌声时不要打开，会严重跑调')
    # 极致音质配置：默认启用特征检索，混合比例0.5
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="model_assets/workspaces/44k/feature_and_index.pkl", help='[极致音质配置] 特征检索索引路径，默认使用feature_and_index.pkl')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=BEST_QUALITY_PRESET["cluster_ratio"], help='[极致音质配置] 特征检索混合比例')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    # 极致音质配置：固定使用 rmvpe
    parser.add_argument('-f0p', '--f0_predictor', type=str, default=BEST_QUALITY_PRESET["f0_predictor"], help='[极致音质配置] 固定使用 rmvpe F0预测器')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='[极致音质配置] 使用浅扩散时自动禁用，无需设置')
    # 极致音质配置：默认启用浅层扩散
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', help='[极致音质配置] 启用浅层扩散（默认已启用）')
    parser.add_argument('--no_shallow_diffusion', action='store_true', help='[极致音质配置] 禁用浅层扩散')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=BEST_QUALITY_PRESET["loudness_envelope_adjustment"], help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络')
    # 极致音质配置：默认启用特征检索
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', help='[极致音质配置] 启用特征检索（默认已启用）')
    parser.add_argument('--no_feature_retrieval', action='store_true', help='[极致音质配置] 禁用特征检索')

    # 极致音质配置：浅扩散设置
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="model_assets/workspaces/44k/diffusion/model_0.pt", help='[极致音质配置] 扩散模型路径')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="model_assets/workspaces/44k/diffusion/config.yaml", help='[极致音质配置] 扩散模型配置文件路径')
    # 极致音质配置：k_step设置为200（极致音质）
    parser.add_argument('-ks', '--k_step', type=int, default=BEST_QUALITY_PRESET["k_step"], help='[极致音质配置] 扩散步数')
    # 极致音质配置：默认启用二次编码
    parser.add_argument('-se', '--second_encoding', action='store_true', help='[极致音质配置] 启用二次编码（默认已启用）')
    parser.add_argument('--no_second_encoding', action='store_true', help='[极致音质配置] 禁用二次编码')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')
    

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=BEST_QUALITY_PRESET["slice_db"], help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=BEST_QUALITY_PRESET["noise_scale"], help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=BEST_QUALITY_PRESET["pad_seconds"], help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default=BEST_QUALITY_PRESET["output_format"], help='音频输出格式')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=BEST_QUALITY_PRESET["linear_gradient_retain"], help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=BEST_QUALITY_PRESET["enhancer_adaptive_key"], help='使增强器适应更高的音域(单位为半音数)|默认为0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=BEST_QUALITY_PRESET["cr_threshold"],help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')


    args = parser.parse_args()

    # 极致音质配置：设置默认值（默认启用，除非明确禁用）
    shallow_diffusion = not args.no_shallow_diffusion
    feature_retrieval = not args.no_feature_retrieval
    second_encoding = not args.no_second_encoding

    clean_names = args.clean_names
    trans = args.trans
    speaker = args.speaker
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    # 极致音质配置：自动设置特征检索路径
    if feature_retrieval and cluster_infer_ratio > 0:
        if args.cluster_model_path == "":
            args.cluster_model_path = "model_assets/workspaces/44k/feature_and_index.pkl"
    else:
        args.cluster_model_path = ""

    # 极致音质配置：使用浅扩散时自动禁用增强器
    if shallow_diffusion:
        enhance = False
    
    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    feature_retrieval)
    
    infer_tool.mkdir(["inference_data/inputs", "inference_data/outputs"])

    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"inference_data/inputs/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        kwarg = {
            "raw_audio_path" : raw_audio_path,
            "spk" : speaker,
            "tran" : tran,
            "slice_db" : slice_db,
            "cluster_infer_ratio" : cluster_infer_ratio,
            "auto_predict_f0" : auto_predict_f0,
            "noice_scale" : noice_scale,
            "pad_seconds" : pad_seconds,
            "clip_seconds" : clip,
            "lg_num": lg,
            "lgr_num" : lgr,
            "f0_predictor" : f0p,
            "enhancer_adaptive_key" : enhancer_adaptive_key,
            "cr_threshold" : cr_threshold,
            "k_step":k_step,
            "second_encoding":second_encoding,
            "loudness_envelope_adjustment":loudness_envelope_adjustment
        }
        audio = svc_model.slice_inference(**kwarg)
        key = "auto" if auto_predict_f0 else f"{tran}key"
        cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
        isdiffusion = "sovits"
        if shallow_diffusion :
            isdiffusion = "sovdiff"
        if only_diffusion :
            isdiffusion = "diff"
        res_path = f'inference_data/outputs/{clean_name}_{key}_{speaker}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
        svc_model.clear_empty()
            
if __name__ == '__main__':
    main()
