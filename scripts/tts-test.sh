#!/usr/bin/env bash
# Quick TTS test: synthesize text and play audio locally.
#
# Usage (CustomVoice — named speakers):
#   ./scripts/tts-test.sh "Hello, this is a test." --voice Vivian
#
# Usage (Base — voice clone from reference audio):
#   ./scripts/tts-test.sh "Hello, this is a test." --ref-audio /path/to/ref.wav
#   ./scripts/tts-test.sh "Hello." --ref-audio /path/to/ref.wav --ref-text "transcript of ref"

set -euo pipefail

API_URL="${TTS_API_URL:-http://localhost:8000/v1/audio/speech}"
VOICE=""
LANGUAGE="Auto"
FORMAT="wav"
TEXT=""
REF_AUDIO=""
REF_TEXT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --voice)     VOICE="$2"; shift 2 ;;
        --language)  LANGUAGE="$2"; shift 2 ;;
        --format)    FORMAT="$2"; shift 2 ;;
        --url)       API_URL="$2"; shift 2 ;;
        --ref-audio) REF_AUDIO="$2"; shift 2 ;;
        --ref-text)  REF_TEXT="$2"; shift 2 ;;
        -*)          echo "Unknown option: $1" >&2; exit 1 ;;
        *)           TEXT="$1"; shift ;;
    esac
done

if [[ -z "$TEXT" ]]; then
    echo "Usage: $0 \"Text to speak\" [--voice NAME | --ref-audio FILE] [--language LANG]" >&2
    exit 1
fi

OUTFILE="$(mktemp /tmp/tts-XXXXXX.${FORMAT})"
trap 'rm -f "$OUTFILE"' EXIT

# Build JSON payload depending on mode
PAYLOAD_FILE="$(mktemp /tmp/tts-payload-XXXXXX.json)"
trap 'rm -f "$OUTFILE" "$PAYLOAD_FILE"' EXIT

if [[ -n "$REF_AUDIO" ]]; then
    # Voice clone mode (Base task) — write base64 to file to avoid arg-list limits
    echo "Requesting TTS (voice clone): language=$LANGUAGE format=$FORMAT"
    echo "Reference: $REF_AUDIO"
    REF_B64="data:audio/wav;base64,$(base64 -w0 "$REF_AUDIO")"
    jq -n \
        --arg input "$TEXT" \
        --arg language "$LANGUAGE" \
        --arg format "$FORMAT" \
        --arg ref_text "$REF_TEXT" \
        '{input: $input, task_type: "Base", language: $language, response_format: $format,
          x_vector_only_mode: true, max_new_tokens: 2048}
         + (if $ref_text != "" then {ref_text: $ref_text} else {} end)' \
        > "$PAYLOAD_FILE"
    # Inject the large ref_audio field via jq --rawfile to avoid shell arg limits
    echo "$REF_B64" > "${PAYLOAD_FILE}.b64"
    jq --rawfile b64 "${PAYLOAD_FILE}.b64" '. + {ref_audio: ($b64 | rtrimstr("\n"))}' "$PAYLOAD_FILE" \
        > "${PAYLOAD_FILE}.tmp" && mv "${PAYLOAD_FILE}.tmp" "$PAYLOAD_FILE"
    rm -f "${PAYLOAD_FILE}.b64"
else
    # CustomVoice mode (named speaker)
    VOICE="${VOICE:-Vivian}"
    echo "Requesting TTS (custom voice): voice=$VOICE language=$LANGUAGE format=$FORMAT"
    jq -n \
        --arg input "$TEXT" \
        --arg voice "$VOICE" \
        --arg language "$LANGUAGE" \
        --arg format "$FORMAT" \
        '{input: $input, voice: $voice, language: $language, response_format: $format}' \
        > "$PAYLOAD_FILE"
fi

echo "Text: $TEXT"
echo "Endpoint: $API_URL"

HTTP_CODE=$(curl -s -w '%{http_code}' -o "$OUTFILE" \
    -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d @"$PAYLOAD_FILE")

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "Error: API returned HTTP $HTTP_CODE" >&2
    cat "$OUTFILE" >&2
    exit 1
fi

FILE_SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || stat -f%z "$OUTFILE" 2>/dev/null)
echo "Received ${FILE_SIZE} bytes -> $OUTFILE"

# Check if response is actually audio (not a JSON error with 200 status)
if file "$OUTFILE" | grep -q "JSON"; then
    echo "Error: Server returned JSON instead of audio:" >&2
    cat "$OUTFILE" >&2
    exit 1
fi

# Play with the best available player
if command -v pw-play &>/dev/null; then
    pw-play "$OUTFILE"
elif command -v ffplay &>/dev/null; then
    ffplay -nodisp -autoexit -loglevel quiet "$OUTFILE"
else
    echo "No audio player found. Audio saved at: $OUTFILE"
    trap - EXIT  # don't delete if we can't play
fi
