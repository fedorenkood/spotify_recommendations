from sklearn.model_selection import train_test_split
import os
import zipfile
import tensorflow as tf
import pandas as pd
from varname import nameof

def main():
    # model.fit(x, y)
    label_to_playlists = {
        'jazz_playlists'      : 1,
        'classical_playlists' : 2,
        'indie_playlists'     : 3,
        'country_playlists'   : 4,
        'rock_playlists'      : 5,
        'pop_playlists'       : 6,
        'hip_hop_playlists'   : 7,
        'edm_playlists'       : 8
    }

    label = 1
    all_features_df = None
    all_labels_df = None

    for name, label in label_to_playlists.items():
        playlists_df = None
        path = os.path.join(os.getcwd(), "data", f"{name}_label_{label}.csv")
        dataframe = pd.read_csv(path)
        dataframe = dataframe.drop_duplicates(subset='track_id', keep=False)
        labels = dataframe.pop('label')
        labels = labels.astype('float32')
        features = dataframe.drop(['Unnamed: 0', 'track_id'], axis=1)
        features.astype('float32')
        if all_features_df is None:
            all_features_df = features
        else:
            all_features_df = pd.concat(
                    [all_features_df, features],
                    axis=0,
                    join="outer",
                    ignore_index=False,
                    keys=None,
                    levels=None,
                    names=None,
                    verify_integrity=False,
                    copy=True,
                )
        if all_labels_df is None:
            all_labels_df = labels
        else:
            all_labels_df = pd.concat(
                    [all_labels_df, labels],
                    axis=0,
                    join="outer",
                    ignore_index=False,
                    keys=None,
                    levels=None,
                    names=None,
                    verify_integrity=False,
                    copy=True,
                )


    x_train, x_test, y_train, y_test = train_test_split(all_features_df, all_labels_df, test_size=0.20)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        steps_per_epoch=10,
        epochs=10,
        validation_steps=5,
        verbose=2)

    print("Model performance on test set = " + str(model.evaluate(
        x=x_test,
        y=y_test
    )))

if __name__ == "__main__":
    main()
