import argparse
import glob
import logging

from dpr.indexer.faiss_indexers import (
    DenseFlatIndexer,
    DenseReconIndexer,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def main(args):    
    logger.info("Building index ...")
    if args.index_type == "flat":
        index = DenseFlatIndexer(args.vector_size, args.index_buffer)
    else:
        index = DenseReconIndexer(args.vector_size, args.index_buffer)
    ctx_files_pattern = args.encoded_ctx_file
    logger.info(ctx_files_pattern)
    input_paths = glob.glob(ctx_files_pattern)
    logger.info(input_paths)

    logger.info("Reading all passages data from files: %s", input_paths)
    index.index_data(input_paths)
    index.serialize(args.index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoded_ctx_file",
        type=str,
        default=None,
        help="Glob path to encoded passages (from generate_dense_embeddings tool)",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path for saving the built index",
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=768,
        help="Encoded vector size",
    )

    parser.add_argument(
        "--index_type",
        type=str,
        default="flat",
        help="index type",
    )

    args = parser.parse_args()
    main(args)
