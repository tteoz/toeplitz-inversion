#!/bin/awk -f
{ gtotal += $2; mtotal += $4 }
END { print gtotal/(NR - 2); print mtotal/(NR - 2) }

