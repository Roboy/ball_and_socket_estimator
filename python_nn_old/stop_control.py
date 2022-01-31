import rospy
import std_msgs.msg

rospy.init_node("stop_control")


tracker_down = False
tracker_down_cnt = 0
last_tracker_down = 0


def tracker_down_callback(data):
    global tracker_down, tracker_down_cnt
    tracker_down = data.data
    if tracker_down:
        rospy.set_param('/publish_cardsflow', False)


tracking_loss_sub = rospy.Subscriber('/tracking_loss', std_msgs.msg.Bool, tracker_down_callback)
rospy.spin()
