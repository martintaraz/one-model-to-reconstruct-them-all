import argparse
from pathlib import Path

from PIL import Image
from pytorch_training.images import make_image

from data.demo_dataset import DemoDataset
from data.demo_dataset_folder import DemoDatasetFolder
from networks import get_autoencoder, load_weights
from utils.config import load_config
from utils.data_loading import build_data_loader


def main(args):
    root_dir = Path(args.autoencoder_checkpoint).parent.parent
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    config = load_config(args.autoencoder_checkpoint, None)
    config['batch_size'] = 1
    autoencoder = get_autoencoder(config).to(args.device)
    autoencoder = load_weights(autoencoder, args.autoencoder_checkpoint, key='autoencoder')
    if not args.folder:
        input_image = Path(args.image)
        data_loader = build_data_loader(input_image, config, config['absolute'], shuffle_off=True, dataset_class=DemoDataset)
    else:
        # load the whole folder into a dataset
        input_folder = Path(args.image)
        data_loader = build_data_loader(input_folder, config, config['absolute'], shuffle_off=True, dataset_class=DemoDatasetFolder)
    for idx, image in enumerate(data_loader):
        image = {k: v.to(args.device) for k, v in image.items()}
        reconstructed = Image.fromarray(make_image(autoencoder(image['input_image'])[0].squeeze(0)))
        # rescale the image to the original dimensions
        reconstructed = reconstructed.resize((1920,1080))
        output_name = Path(args.output_dir) / f"reconstructed_{idx}_stylegan_{config['stylegan_variant']}_{'w_only' if config['w_only'] else 'w_plus'}.png"
        reconstructed.save(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="reconstruct a given image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("autoencoder_checkpoint", help='Path to autoencoder checkpoint which shall be used for embedding')
    parser.add_argument("image", help="image to reconstruct")
    parser.add_argument("--device", default='cpu', help="which device to use (cuda, or cpu)")
    parser.add_argument("--output-dir", default='.')
    parser.add_argument("--folder", default=False)

    main(parser.parse_args())
