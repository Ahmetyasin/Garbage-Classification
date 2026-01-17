"""
Web scraping script for collecting additional garbage classification images.
Uses icrawler to scrape images from Bing.
"""

import os
import argparse
from pathlib import Path
from icrawler.builtin import BingImageCrawler
from tqdm import tqdm
import yaml

# Search queries for each class
SEARCH_QUERIES = {
    "cardboard": [
        "cardboard waste",
        "cardboard box garbage",
        "recyclable cardboard"
    ],
    "glass": [
        "glass bottle waste",
        "broken glass garbage",
        "glass recycling"
    ],
    "metal": [
        "metal can waste",
        "aluminum can garbage",
        "metal recycling"
    ],
    "paper": [
        "paper waste",
        "newspaper garbage",
        "paper recycling"
    ],
    "plastic": [
        "plastic bottle waste",
        "plastic container garbage",
        "plastic recycling"
    ],
    "trash": [
        "general waste",
        "non-recyclable garbage",
        "landfill waste",
        "mixed garbage"
    ]
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def scrape_images_for_class(
    class_name: str,
    output_dir: str,
    num_images: int,
    queries: list
) -> int:
    """
    Scrape images for a specific class using multiple search queries.

    Args:
        class_name: Name of the garbage class
        output_dir: Directory to save images
        num_images: Target number of images to scrape
        queries: List of search queries for this class

    Returns:
        Number of images successfully scraped
    """
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Calculate images per query
    images_per_query = max(1, num_images // len(queries))
    remaining = num_images - (images_per_query * len(queries))

    total_scraped = 0

    for i, query in enumerate(queries):
        # Add extra images to first query if there's remainder
        target = images_per_query + (remaining if i == 0 else 0)

        print(f"  Scraping '{query}' ({target} images)...")

        # Create crawler with storage in class directory
        crawler = BingImageCrawler(
            storage={'root_dir': str(class_dir)},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
            log_level=50  # Suppress most logging
        )

        # Crawl images
        crawler.crawl(
            keyword=query,
            max_num=target,
            min_size=(100, 100),  # Minimum image size
            file_idx_offset=total_scraped  # Offset to avoid overwriting
        )

        # Count new images
        current_count = len(list(class_dir.glob('*.[jJ][pP][gG]')) +
                          list(class_dir.glob('*.[jJ][pP][eE][gG]')) +
                          list(class_dir.glob('*.[pP][nN][gG]')))
        total_scraped = current_count

    return total_scraped


def scrape_all_classes(config_path: str = "configs/config.yaml") -> dict:
    """
    Scrape images for all garbage classes.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with scraping statistics per class
    """
    config = load_config(config_path)
    output_dir = config['data']['scraped_dir']
    images_per_class = config['scraping']['images_per_class']

    print("="*60)
    print("Starting Web Scraping for Dataset Expansion")
    print("="*60)

    stats = {}

    for class_name in tqdm(images_per_class.keys(), desc="Scraping classes"):
        num_images = images_per_class[class_name]
        queries = SEARCH_QUERIES.get(class_name, [f"{class_name} waste"])

        print(f"\nClass: {class_name} (target: {num_images} images)")

        scraped = scrape_images_for_class(
            class_name=class_name,
            output_dir=output_dir,
            num_images=num_images,
            queries=queries
        )

        stats[class_name] = {
            'target': num_images,
            'scraped': scraped
        }

        print(f"  Completed: {scraped} images scraped")

    print("\n" + "="*60)
    print("Scraping Summary")
    print("="*60)

    total_target = 0
    total_scraped = 0

    for class_name, data in stats.items():
        total_target += data['target']
        total_scraped += data['scraped']
        print(f"{class_name:12s}: {data['scraped']:4d}/{data['target']:4d} images")

    print("-"*60)
    print(f"{'Total':12s}: {total_scraped:4d}/{total_target:4d} images")
    print("="*60)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Scrape images for garbage classification")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--class-name',
        type=str,
        default=None,
        help='Scrape only a specific class (optional)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Override number of images to scrape per class'
    )

    args = parser.parse_args()

    if args.class_name:
        # Scrape single class
        config = load_config(args.config)
        output_dir = config['data']['scraped_dir']
        num_images = args.num_images or config['scraping']['images_per_class'].get(args.class_name, 50)
        queries = SEARCH_QUERIES.get(args.class_name, [f"{args.class_name} waste"])

        print(f"Scraping {num_images} images for class: {args.class_name}")
        scraped = scrape_images_for_class(
            class_name=args.class_name,
            output_dir=output_dir,
            num_images=num_images,
            queries=queries
        )
        print(f"Completed: {scraped} images scraped")
    else:
        # Scrape all classes
        scrape_all_classes(args.config)


if __name__ == "__main__":
    main()
