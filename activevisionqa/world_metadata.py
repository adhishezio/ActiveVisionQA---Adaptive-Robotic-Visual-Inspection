"""Publish the exact generated simulation configuration, never inspection answers."""


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init(args=args)
    node = Node("activevisionqa_workcell_metadata")
    node.declare_parameter("metadata_json", "{}")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    publisher = node.create_publisher(String, "/activevisionqa/workcell", qos)
    payload = str(node.get_parameter("metadata_json").value)

    def publish():
        publisher.publish(String(data=payload))

    node.create_timer(1.0, publish)
    publish()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
