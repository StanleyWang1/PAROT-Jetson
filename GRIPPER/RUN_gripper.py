import time
import numpy as np
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite

from control_table import *
from dynamixel_controller import DynamixelController

# PORT_NAME = "/dev/ttyUSB0"
PORT_NAME = "COM8"

# ----------------------
# Motor IDs
# ----------------------
MOTOR24 = 24
MOTOR25 = 25
MOTORS = (MOTOR24, MOTOR25)

# ----------------------
# Home & Soft Limits (adjust to your robot)
# ----------------------
OFFSET = 0
MOTOR24_HOME -= OFFSET   # gripper "open"
MOTOR25_HOME += OFFSET   # gripper "open"

SOFT_MIN = -2_147_483_648      # signed 32-bit ticks (Extended Position)
SOFT_MAX =  2_147_483_647

def clamp(v, vmin=SOFT_MIN, vmax=SOFT_MAX):
    return max(vmin, min(vmax, int(v)))

# ----------------------
# Operating Mode values (X-series)
# 4  = Extended Position
# 16 = PWM
# ----------------------
MODE_EXTENDED_POSITION = 4
MODE_PWM = 16

# 50% PWM target. On most X-series, PWM limit default ≈ 885 counts.
# 50% ≈ 0.5 * 885 = 443. Adjust if you’ve changed PWM Limit.
PWM_DRIVE = 600

def dynamixel_connect():
    controller = DynamixelController(PORT_NAME, 57600, 2.0)
    # controller = DynamixelController("COM8", 57600, 2.0)

    # GroupSyncWrite for GOAL_POSITION (used in Extended Position mode)
    gsw_pos = GroupSyncWrite(
        controller.port_handler,
        controller.packet_handler,
        GOAL_POSITION[0],
        GOAL_POSITION[1],
    )

    # Reboot motors to a clean state
    for motor_id in MOTORS:
        dxl_comm_result, dxl_error = controller.packet_handler.reboot(
            controller.port_handler, motor_id
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[WARN] Reboot {motor_id} failed: "
                  f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"[WARN] Reboot {motor_id} error: "
                  f"{controller.packet_handler.getRxPacketError(dxl_error)}")
        else:
            print(f"[OK] Motor {motor_id} rebooted")
    time.sleep(2.0)

    # Start in Extended Position mode with torque ON
    set_operating_mode(controller, MODE_EXTENDED_POSITION, torque_cycle=True)
    set_motion_profiles(controller, vel=200, acc=50)

    return controller, gsw_pos

def set_motion_profiles(controller, vel=200, acc=50):
    """Set velocity/accel profiles (ticks/s and ticks/s^2)."""
    for m in MOTORS:
        controller.write(m, PROFILE_VELOCITY, int(vel))
        controller.write(m, PROFILE_ACCELERATION, int(acc))
    time.sleep(0.05)

def set_operating_mode(controller, mode, torque_cycle=True):
    """
    Safely switch operating mode:
    - torque off (if torque_cycle)
    - write operating mode
    - torque on (if torque_cycle)
    """
    if torque_cycle:
        for m in MOTORS:
            controller.write(m, TORQUE_ENABLE, 0)
        time.sleep(0.05)

    for m in MOTORS:
        controller.write(m, OPERATING_MODE, int(mode))
    time.sleep(0.05)

    if torque_cycle:
        for m in MOTORS:
            controller.write(m, TORQUE_ENABLE, 1)
        time.sleep(0.05)

def go_home(controller, gsw_pos):
    """Drive both motors to their home (open) positions in Extended Position mode."""
    pos24 = clamp(MOTOR24_HOME)
    pos25 = clamp(MOTOR25_HOME)

    ok = gsw_pos.addParam(MOTOR24, int(pos24).to_bytes(4, "little", signed=True))
    ok &= gsw_pos.addParam(MOTOR25, int(pos25).to_bytes(4, "little", signed=True))
    if not ok:
        print("[ERROR] GroupSyncWrite addParam failed (home)")
        gsw_pos.clearParam()
        return False

    dxl_comm_result = gsw_pos.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] GroupSyncWrite txPacket (home): "
              f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        gsw_pos.clearParam()
        return False

    gsw_pos.clearParam()
    return True

def pwm_drive(controller, pwm24, pwm25):
    """
    Command PWM directly. Typical safe range is -PWM_LIMIT..+PWM_LIMIT.
    Positive sign is model/installation dependent (may open/close).
    """
    controller.write(MOTOR24, GOAL_PWM, int(pwm24))
    controller.write(MOTOR25, GOAL_PWM, int(pwm25))
    # short settle
    time.sleep(0.02)

def torque_off(controller):
    for m in MOTORS:
        controller.write(m, TORQUE_ENABLE, 0)
    time.sleep(0.05)

def main():
    controller, gsw_pos = dynamixel_connect()
    print("[INFO] Motors initialized in Extended Position mode. Homing...")
    go_home(controller, gsw_pos)
    time.sleep(1.0)
    print("[OK] At home (open).")

    print("\nCommands:")
    print("  g = switch to PWM mode and drive both grippers at ~50% PWM")
    print("  o = switch to Extended Position mode and go to HOME (open)")
    print("  s = torque OFF (disable)")
    print("  q = quit\n")

    try:
        while True:
            cmd = input("Enter command [g/o/s/q]: ").strip().lower()

            if cmd == "g":
                print("[INFO] Switching to PWM mode...")
                set_operating_mode(controller, MODE_PWM, torque_cycle=True)
                # If you need mirrored directions, flip one sign:
                pwm24 = +PWM_DRIVE
                pwm25 = -PWM_DRIVE
                pwm_drive(controller, pwm24, pwm25)
                print(f"[OK] PWM mode. Commanded ~50%: M24={pwm24}, M25={pwm25}")

            elif cmd == "o":
                print("[INFO] Switching to Extended Position mode and homing...")
                set_operating_mode(controller, MODE_EXTENDED_POSITION, torque_cycle=True)
                set_motion_profiles(controller, vel=200, acc=50)
                if go_home(controller, gsw_pos):
                    print("[OK] Back at home (open).")

            elif cmd == "s":
                print("[INFO] Torque OFF for both grippers.")
                torque_off(controller)
                print("[OK] Grippers disabled (torque off).")

            elif cmd == "q":
                print("[INFO] Quitting...")
                break

            else:
                print("[WARN] Unknown command. Use g/o/s/q.")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    finally:
        try:
            torque_off(controller)
        except Exception as e:
            print(f"[WARN] Failed to disable torque cleanly: {e}")
        print("[DONE] Motors stopped and torque off.")

if __name__ == "__main__":
    main()
