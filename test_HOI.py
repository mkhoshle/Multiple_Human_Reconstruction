import os
import sys
from HumanObj_videos_ResNet.lib.config import ConfigContext, parse_args, args
from HumanObj_videos_ResNet.predict.image import Image_processor

CUDA_LAUNCH_BLOCKING = 1
ConfigContext.parsed_args = parse_args(["--configs_yml=configs/image.yml",
                                        '--inputs=demo/videos/demo_frames',
                                        '--output_dir=demo/video_results2',
                                        '--renderer=pytorch3d'])

# second, run the code
processor = Image_processor(args_set=args())
inputs = args().inputs


results = processor.run(inputs)

# from IPython.display import Image, display
# import glob
# for img_path in glob.glob('demo/video_results2/*.jpg'):
#     display(Image(img_path))
