# AnalogVision

AnalogVision is a local web app for converting videos into retro styles and comparing the result against the original.

It supports:

- True VHS conversion
- Noir black-and-white conversion
- Smart 4:3 formatting
- Optional degraded audio
- Before/after curtain view
- Before/after split view
- MP4 download
- Short before/after GIF download
- Progress and time estimate while converting

## Demo

### VHS Conversion

![VHS conversion demo](Media/vhs.gif)

### Noir B&W Conversion

![Noir B&W conversion demo](Media/noir.gif)

## How To Run

Install Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the app from the project folder:

```bash
python web_app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How To Use

1. Drop a video into the upload area or click to browse.
2. Choose `True VHS` or `Noir B&W`.
3. Toggle Smart 4:3 and degraded audio.
4. Click `Convert`.
5. Compare the original and converted video.
6. Download the MP4 or GIF.

## GitHub Notes

Generated videos and local conversion files are ignored by Git.

GIF files are not ignored, so the demo files in `Media/` can be added to the repository.

## License

MIT
