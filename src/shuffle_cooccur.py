"""
Python-based shuffle for GloVe cooccurrence files.
Handles large files that cause GloVe's C shuffle to segfault.
"""
import struct
import random
import argparse
import sys
from tqdm import tqdm

# CREC structure: two ints (word1, word2) and one float (cooccurrence value)
CREC_SIZE = struct.calcsize('iif')  # 12 bytes: int(4) + int(4) + float(4)

def shuffle_cooccur(input_file, output_file, memory_mb=4096, verbose=2):
    """
    Shuffle GloVe cooccurrence binary file.
    
    Args:
        input_file: Path to input cooccurrence.bin file
        output_file: Path to output shuffled file
        memory_mb: Memory limit in MB for buffering
        verbose: Verbosity level
    """
    if verbose >= 2:
        print(f"Reading cooccurrence file: {input_file}")
    
    # Read all records into memory (or in chunks if too large)
    records = []
    buffer_size = memory_mb * 1024 * 1024  # Convert MB to bytes
    
    try:
        with open(input_file, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Reset to beginning
            
            total_records = file_size // CREC_SIZE
            if verbose >= 2:
                print(f"File size: {file_size:,} bytes")
                print(f"Total records: {total_records:,}")
            
            # Read entire file (Python can handle large files in memory)
            # If file is too large, we'll read in chunks but still load all records
            if file_size > buffer_size:
                if verbose >= 2:
                    print(f"File is large ({file_size / (1024**3):.2f} GB), reading in chunks...")
                
                # Read in chunks and parse records
                records = []
                bytes_read = 0
                while bytes_read < file_size:
                    chunk = f.read(min(buffer_size, file_size - bytes_read))
                    if not chunk:
                        break
                    
                    # Parse chunk into records
                    num_records_in_chunk = len(chunk) // CREC_SIZE
                    for i in range(num_records_in_chunk):
                        offset = i * CREC_SIZE
                        record = struct.unpack('iif', chunk[offset:offset+CREC_SIZE])
                        records.append(record)
                    
                    bytes_read += len(chunk)
                    if verbose >= 2 and len(records) % 1000000 == 0:
                        print(f"  Read {len(records):,} records ({bytes_read / file_size * 100:.1f}%)...")
            else:
                # Read entire file
                data = f.read()
                num_records = len(data) // CREC_SIZE
                records = []
                for i in tqdm(range(num_records), desc="Reading records", disable=verbose<2):
                    offset = i * CREC_SIZE
                    record = struct.unpack('iif', data[offset:offset+CREC_SIZE])
                    records.append(record)
        
        if verbose >= 2:
            print(f"Loaded {len(records):,} records")
            print("Shuffling records...")
        
        # Shuffle
        random.shuffle(records)
        
        if verbose >= 2:
            print(f"Writing shuffled file: {output_file}")
        
        # Write shuffled records
        with open(output_file, 'wb') as f:
            for record in tqdm(records, desc="Writing shuffled records", disable=verbose<2):
                f.write(struct.pack('iif', *record))
        
        if verbose >= 2:
            print(f"Successfully shuffled {len(records):,} records")
        
        return True
        
    except Exception as e:
        print(f"Error during shuffle: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle GloVe cooccurrence binary file')
    parser.add_argument('input_file', type=str, help='Input cooccurrence.bin file')
    parser.add_argument('output_file', type=str, help='Output shuffled file')
    parser.add_argument('-memory', type=float, default=4096.0, help='Memory limit in MB (default: 4096)')
    parser.add_argument('-verbose', type=int, default=2, help='Verbosity level (default: 2)')
    
    args = parser.parse_args()
    
    success = shuffle_cooccur(args.input_file, args.output_file, args.memory, args.verbose)
    sys.exit(0 if success else 1)

