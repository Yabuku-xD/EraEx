import json
import zstandard as zstd
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import polars as pl
from datetime import date

NDJSON_FIELD_MAPPING = {
    "id": "soundcloud_id",
    "title": "title",
    "user.username": "artist",
    "genre": "genre",
    "tag_list": "tags",
    "description": "description",
    "playback_count": "playback_count",
    "permalink_url": "permalink_url",
    "created_at": "created_at",
}


def extract_nested_field(obj: dict, path: str) -> Any:
    parts = path.split(".")
    current = obj
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def parse_ndjson_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    mapped = {}
    for source_field, target_field in NDJSON_FIELD_MAPPING.items():
        value = extract_nested_field(obj, source_field)
        mapped[target_field] = value
    
    if mapped.get("created_at"):
        try:
            year_str = mapped["created_at"][:4]
            mapped["year"] = int(year_str)
        except (ValueError, TypeError):
            mapped["year"] = None
    else:
        mapped["year"] = None
    
    mapped["track_id"] = str(mapped.get("soundcloud_id", "")) if mapped.get("soundcloud_id") else None
    mapped["extracted_vibe_text"] = obj.get("caption") or obj.get("label_name") or ""
    mapped["ingest_date"] = date.today()
    
    return mapped


def stream_ndjson_zst(file_path: Path, chunk_size: int = 10000) -> Iterator[list]:
    file_path = Path(file_path)
    
    dctx = zstd.ZstdDecompressor()
    
    with open(file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            chunk = []
            
            while True:
                data = reader.read(1024 * 1024)
                if not data:
                    break
                
                buffer += data
                lines = buffer.split(b"\n")
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    if not line.strip():
                        continue
                    
                    parsed = parse_ndjson_line(line.decode('utf-8', errors='ignore'))
                    if parsed:
                        chunk.append(parsed)
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            
            if buffer.strip():
                parsed = parse_ndjson_line(buffer.decode('utf-8', errors='ignore'))
                if parsed:
                    chunk.append(parsed)
            
            if chunk:
                yield chunk


def stream_plain_ndjson(file_path: Path, chunk_size: int = 10000) -> Iterator[list]:
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
        chunk = []
        for line in fh:
            if not line.strip():
                continue
            
            parsed = parse_ndjson_line(line)
            if parsed:
                chunk.append(parsed)
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        if chunk:
            yield chunk


def ingest_ndjson_to_parquet(
    input_path: Path,
    output_dir: Path,
    year_range: range = range(2012, 2019),
    chunk_size: int = 100000,
    max_rows: int = None
) -> dict:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {year: 0 for year in year_range}
    stats["skipped"] = 0
    stats["total"] = 0
    
    year_buffers = {year: [] for year in year_range}
    buffer_size = 50000
    
    if input_path.suffix == ".zst" or str(input_path).endswith(".ndjson.zst"):
        stream_func = stream_ndjson_zst
    else:
        stream_func = stream_plain_ndjson
    
    print(f"Reading from: {input_path}")
    print(f"Output directory: {output_dir}")
    
    batch_num = 0
    total_processed = 0
    
    for chunk in stream_func(input_path, chunk_size=10000):
        for record in chunk:
            year = record.get("year")
            
            if year is None or year not in year_range:
                stats["skipped"] += 1
                continue
            
            year_buffers[year].append(record)
            stats[year] += 1
            stats["total"] += 1
            total_processed += 1
            
            if max_rows and total_processed >= max_rows:
                break
        
        for year in year_range:
            if len(year_buffers[year]) >= buffer_size:
                batch_num += 1
                _flush_buffer(year_buffers[year], output_dir, year, batch_num)
                year_buffers[year] = []
        
        if total_processed % 100000 == 0:
            print(f"  Processed: {total_processed:,} rows")
        
        if max_rows and total_processed >= max_rows:
            print(f"  Reached max_rows limit: {max_rows:,}")
            break
    
    for year in year_range:
        if year_buffers[year]:
            batch_num += 1
            _flush_buffer(year_buffers[year], output_dir, year, batch_num)
    
    return stats


def _flush_buffer(records: list, output_dir: Path, year: int, batch_num: int):
    if not records:
        return
    
    year_dir = output_dir / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    
    df = pl.DataFrame(records)
    
    if "ingest_date" in df.columns:
        df = df.with_columns([
            pl.col("ingest_date").cast(pl.Date)
        ])
    
    out_path = year_dir / f"batch_{batch_num:05d}.parquet"
    df.write_parquet(out_path)
