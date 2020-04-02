import rospy
import numpy as np
from std_msgs.msg import Int16

rospy.init_node("motor_test")

command = []
command.append(rospy.Publisher('/motor_command/motor0', Int16, queue_size=1))
command.append(rospy.Publisher('/motor_command/motor1', Int16, queue_size=1))
command.append(rospy.Publisher('/motor_command/motor2', Int16, queue_size=1))

rate = rospy.Rate(10)

pos = []
# pos.append(np.linspace(120,270,100))
# pos.append(np.linspace(170,170,100))
# pos.append(np.linspace(97,97,100))

# pos.append(np.linspace(190,190,100))
# pos.append(np.linspace(200,90,100))
# pos.append(np.linspace(97,97,100))

# pos.append(np.linspace(190,190,100))
# pos.append(np.linspace(170,170,100))
# pos.append(np.linspace(60,120,100))

pos.append(np.concatenate([np.linspace(90,270,100),np.linspace(190,190,100),np.linspace(190,190,100)]))
pos.append(np.concatenate([np.linspace(190,190,100),np.linspace(200,90,100),np.linspace(190,190,100)]))
pos.append(np.concatenate([np.linspace(95,95,100),np.linspace(95,95,100),np.linspace(50,120,100)]))

for i in range(0,len(pos[0])):
    if rospy.is_shutdown():
        break
    for j in range(0,3):
        msg = Int16()
        msg.data = int(pos[j][i])
        print(msg.data)
        command[j].publish(msg)
    rate.sleep()
msg = Int16()
msg.data = 190
command[0].publish(msg)
msg.data = 170
command[1].publish(msg)
msg.data = 95
command[2].publish(msg)
rospy.spin()
