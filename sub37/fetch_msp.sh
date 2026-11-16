#!/bin/sh
# The Sub 37 knob map (hardware-derived NRPN/SysEx field offsets) from bsp2/tks, pinned. The repo declares no
# licence, so it is fetched rather than vendored.  usage: sh fetch_msp.sh [dest]
SHA=fa9455bde750b9e84fc25750b8af29f8d55068da
curl -sfL "https://raw.githubusercontent.com/bsp2/tks/$SHA/tks-projects/apps/synergy/midi_synth_profiles/moog_sub37.msp" -o "${1:-moog_sub37.msp}"
