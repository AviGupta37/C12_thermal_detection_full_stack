# Aerial Thermal Human Detection System (Skydroid C12 & PX4)

This workspace contains the complete production ground station software, YOLO fine-tuning pipeline, dataset tools, and tests for aerial thermal human detection using the Skydroid C12 camera and PX4 flight controller.

The complete codebase and documentation are organized in `c12_thermal_detection/`.

## Quick Start

1. **Launch Desktop Ground Station UI**:
   ```bash
   python main.py
   ```
   Or inside `c12_thermal_detection`:
   ```bash
   cd c12_thermal_detection
   python main.py
   ```

2. **Run Headless Smoke Verification**:
   ```bash
   python main.py --headless
   ```

3. **Run Unit Tests**:
   ```bash
   pytest c12_thermal_detection/tests/ -v
   ```

4. **Detailed Documentation**:
   - [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
   - [System Architecture & User Manual](c12_thermal_detection/README.md)

