#ifndef MOTOR_H
#define MOTOR_H

class MotorController {
private:
    int leftForward;
    int leftBackward;
    int rightForward;
    int rightBackward;
    int leftPWM;
    int rightPWM;

public:
    MotorController(int lf, int lb, int rf, int rb, int lpwm, int rpwm) {
        leftForward = lf;
        leftBackward = lb;
        rightForward = rf;
        rightBackward = rb;
        leftPWM = lpwm;
        rightPWM = rpwm;

        pinMode(leftForward, OUTPUT);
        pinMode(leftBackward, OUTPUT);
        pinMode(rightForward, OUTPUT);
        pinMode(rightBackward, OUTPUT);
        pinMode(leftPWM, OUTPUT);
        pinMode(rightPWM, OUTPUT);
    }

    void forward(int speed = 255) {
        digitalWrite(leftForward, HIGH);
        digitalWrite(leftBackward, LOW);
        digitalWrite(rightForward, HIGH);
        digitalWrite(rightBackward, LOW);
        analogWrite(leftPWM, speed);
        analogWrite(rightPWM, speed);
    }

    void backward(int speed = 255) {
        digitalWrite(leftForward, LOW);
        digitalWrite(leftBackward, HIGH);
        digitalWrite(rightForward, LOW);
        digitalWrite(rightBackward, HIGH);
        analogWrite(leftPWM, speed);
        analogWrite(rightPWM, speed);
    }

    void turnLeft(int speed = 200) {
        digitalWrite(leftForward, LOW);
        digitalWrite(leftBackward, HIGH);
        digitalWrite(rightForward, HIGH);
        digitalWrite(rightBackward, LOW);
        analogWrite(leftPWM, speed);
        analogWrite(rightPWM, speed);
    }

    void turnRight(int speed = 200) {
        digitalWrite(leftForward, HIGH);
        digitalWrite(leftBackward, LOW);
        digitalWrite(rightForward, LOW);
        digitalWrite(rightBackward, HIGH);
        analogWrite(leftPWM, speed);
        analogWrite(rightPWM, speed);
    }

    void stop() {
        digitalWrite(leftForward, HIGH);
        digitalWrite(leftBackward, HIGH);
        digitalWrite(rightForward, HIGH);
        digitalWrite(rightBackward, HIGH);
        analogWrite(leftPWM, 0);
        analogWrite(rightPWM, 0);
    }

    void processCommand(String command, int speed = 200) {
        if (command == "forward" || command == "F") {
            forward(speed);
        } else if (command == "backward" || command == "B") {
            backward(speed);
        } else if (command == "left" || command == "L") {
            turnLeft(speed);
        } else if (command == "right" || command == "R") {
            turnRight(speed);
        } else if (command == "stop" || command == "S") {
            stop();
        }
    }
};

#endif
