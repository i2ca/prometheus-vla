from evdev import UInput, ecodes as e, AbsInfo

cap = {
    e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y],
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(0, -32768, 32767, 0, 0, 0)),
        (e.ABS_Y, AbsInfo(0, -32768, 32767, 0, 0, 0)),
    ],
}

ui = UInput(cap, name="Virtual Xbox Controller")

print("Joystick virtual criado!")
input("Pressione Enter para sair...")

