"""Transfer-learning image classifier using TensorFlow directory datasets."""
import argparse
import tensorflow as tf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()
    train = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="training", seed=42,
        image_size=(args.image_size,args.image_size), batch_size=32)
    val = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="validation", seed=42,
        image_size=(args.image_size,args.image_size), batch_size=32)
    classes = train.class_names
    base = tf.keras.applications.MobileNetV2(input_shape=(args.image_size,args.image_size,3),
        include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input((args.image_size,args.image_size,3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs)
    model.save("food_classifier.keras")
    with open("classes.txt","w",encoding="utf-8") as f:
        f.write("\n".join(classes))
    print("Saved food_classifier.keras; classes:", classes)

if __name__ == "__main__":
    main()
