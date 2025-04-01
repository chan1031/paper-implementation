# config.py

class Config:
    # Hyperparameters
    batch_size = 12
    iterations = 1000
    learning_rate = 0.002
    l2_weight = 0.001
    target = 463  # 예: 463 - broom

    # Environment settings
    logdir = 'logdir'
    model_dir = 'model_dir'
    model_url = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
    model_name = 'inception_v3.ckpt'
    image_dir = 'image_dir'

    # Renderer settings
    obj = '3d_model/barrel.obj'
    texture = '3d_model/barrel.jpg'
    camera_distance_min = 1.8
    camera_distance_max = 2.3
    x_translation_min = -0.05
    x_translation_max = 0.05
    y_translation_min = -0.05
    y_translation_max = 0.05

    # Post-processing settings
    print_error = True
    photo_error = True
    background_min = 0.1
    background_max = 1.0
    light_add_min = -0.15
    light_add_max = 0.15
    light_mult_min = 0.5
    light_mult_max = 2.0
    channel_add_min = -0.15
    channel_add_max = 0.15
    channel_mult_min = 0.7
    channel_mult_max = 1.3
    stddev = 0.1
    
    save_adv = True
    save_adv_path = './adv_images'

# cfg 객체를 생성해서 사용
cfg = Config()
