# Task 1.2: Dataset Audit Summary

### Class Counts

| Class     |   Count | Percentage   |
|:----------|--------:|:-------------|
| AiArtData |     539 | 55.28%       |
| RealArt   |     436 | 44.72%       |

### Metric Distributions (Min, Median, Max)

| Subset    |   Count | Width (Min/Med/Max)   | Height (Min/Med/Max)   | Size KB (Min/Med/Max)   |
|:----------|--------:|:----------------------|:-----------------------|:------------------------|
| Overall   |     975 | 183 / 1024 / 8495     | 135 / 768 / 7091       | 11.0 / 165.1 / 9956.4   |
| AiArtData |     539 | 225 / 1024 / 6000     | 135 / 800 / 4001       | 11.5 / 195.1 / 5494.3   |
| RealArt   |     436 | 183 / 920 / 8495      | 158 / 720 / 7091       | 11.0 / 141.4 / 9956.4   |

### Integrity and Duplication

- **Total Scanned**: 975
- **Valid Images**: 975
- **Corrupt Files**: 0
- **Exact Duplicates (SHA-256 match)**: 2

### Color Modes

| class     |   L |   P |   RGB |   RGBA |
|:----------|----:|----:|------:|-------:|
| AiArtData |   1 |  14 |   489 |     35 |
| RealArt   |   6 |   9 |   407 |     14 |

### Formats

| class     |   JPEG |   PNG |
|:----------|-------:|------:|
| AiArtData |    429 |   110 |
| RealArt   |    403 |    33 |
