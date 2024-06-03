import paho.mqtt.client as mqtt

# MQTT broker details
broker_address = "broker.emqx.io"
broker_port = 1883
topic = "test"

# Callback function for when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribe to the topic
    client.subscribe(topic)

# Callback function for when a message is received
def on_message(client, userdata, msg):
    print("Received message: "+msg.payload.decode())

# Create an MQTT client instance
client = mqtt.Client()

# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, broker_port, 60)

# Start the MQTT client loop
client.loop_start()

# Publish a message
client.publish(topic, "Hello, MQTTX!")

# Keep the script running to receive messages
try:
    while True:
        pass
except KeyboardInterrupt:
    # Disconnect from the MQTT broker
    client.loop_stop()
    client.disconnect()
