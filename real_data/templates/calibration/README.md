# Calibration

File: `calibration.json`

Keep the exact calibration metadata used for the session:

- UR calibration file path
- camera frame
- robot base frame
- end-effector frame
- camera intrinsics path
- hand-eye transform path
- validation pose error

The pipeline does not trust real robot data unless calibration provenance is
kept with the session.
