# Drawing Review Agent

Compares IEM schematic and wiring diagram PDFs — BOM tables, 3L circuit diagrams, and WD wiring analysis.

## Setup

```bash
pip install -r iem_bom_agent/requirements.txt
```

Requires:
- **OCR server** running on `localhost:8080` (SGLang with `zai-org/GLM-OCR`)
- **AWS credentials** configured for Bedrock (`us-east-1`) and S3

Deploy model,

```
SGLANG_ENABLE_SPEC_V2=1 sglang serve --model-path zai-org/GLM-OCR --port 8080 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --served-model-name glm-ocr
```

## Run as API Server

```bash
uvicorn drawing_analyzer_agent.api:app --host 0.0.0.0 --port 8000 --workers 4
```

Docs at `http://localhost:8000/docs`

### Submit a job

```bash
curl -s -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "schematic_path": "s3://bucket/124329-11_R3.pdf",
    "wiring_diagram_path": "s3://bucket/124329-11WD_R0.pdf",
    "output_s3_path": "s3://bucket/reports/",
    "mode": "bom"
  }' | python3 -m json.tool
```

Modes: `bom`, `circuit`, `both`, `wdanalysis`

### Poll status

```bash
curl -s http://localhost:8000/jobs/{job_id} | python3 -m json.tool
```

### Get live events (for UI)

```bash
curl -s "http://localhost:8000/jobs/{job_id}/events?since_step=0" | python3 -m json.tool
```

### Download report

```bash
curl -o report.html http://localhost:8000/reports/{job_id}/bom_comparison_report.html
```

## Run as CLI

```bash
python3 -m drawing_analyzer_agent.agent.cli \
  --schematic docs-sample/124329-11_R3.pdf \
  --wiring-diagram docs-sample/124329-11WD_R0.pdf \
  --mode bom -v
```
