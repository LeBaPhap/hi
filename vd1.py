import paho.mqtt.client as mqtt

# Define the MQTT broker details
broker = "broker.emqx.io"
port = 1883
topic = "test/topic"

# Define the callback function for when a message is received
def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()} on topic {message.topic}")
    print(f"Received message: {message.payload.decode()}")
# Create an MQTT client instance
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Assign the on_message callback function
client.on_message = on_message

# Connect to the broker
client.connect(broker, port, 60)

# Subscribe to the topic
client.subscribe(topic)

# Start the loop to process received messages
client.loop_forever()
