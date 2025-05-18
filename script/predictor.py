import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Flatten
import geopandas as gpd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def read_and_precess_dataset_weather(datapath, feadturepath, targetfactor, predictor):
    X_df = pd.read_csv(datapath)
    X_df['datetime'] = pd.to_datetime(X_df['date'])
    # shift the data by one day
    X_df['datetime'] = X_df['datetime'] - pd.DateOffset(days=1)

    feature_df = pd.read_csv(feadturepath)
    feature_df['datetime'] = pd.to_datetime(feature_df['datetime_date'])
    #change the datetime column name to date
    feature_df = feature_df.rename(columns={'datetime':'date'})

    grouped_by_date = X_df.groupby("date")

    grid_height = 31
    grid_width = 33
    reshaped_data = []

    for date, group in grouped_by_date:

        reshaped_grid = group[targetfactor].to_numpy().reshape(grid_height, grid_width, -1)
        # flattened_grid = reshaped_grid.flatten()  # 展平为一维数组

        # 将数据添加到列表中
        reshaped_data.append((date, reshaped_grid))

        # reshaped_data.append(group["keep_length"].to_numpy().reshape(grid_height, grid_width))

    # Convert the list of tuples into a DataFrame
    reshaped_df = pd.DataFrame(reshaped_data, columns=['date', 'runner_matrix'])

    # Ensure the date column is the right type
    reshaped_df['date'] = pd.to_datetime(reshaped_df['date'])

    # Convert the 'date' column to datetime format if it's not already
    feature_df['date'] = pd.to_datetime(feature_df['datetime_date'])  
    feature_df.drop('datetime_date', axis=1, inplace=True)  # remove the old date column

    # merge runner matrix with weather data
    combined_data = pd.merge(reshaped_df, feature_df, on='date')

    # We need to create sequences -- let's start by initializing our lists
    sequences = []
    next_day_runner_matrix = []

    for i in range(len(combined_data) - 7):
        # Get the last 7 days + current day to predict next day
        last_7_days = combined_data.iloc[i:i+7]
        
        # Sequence of runner matrices
        runner_sequence = np.stack(last_7_days['runner_matrix'].to_numpy()) 
        
        # Sequence of additional features for the last 7 days
        features_sequence = last_7_days.drop(columns=['date', 'runner_matrix']).to_numpy()
        # Reshape to have the same second and third dimensions as the runner_sequence
        features_sequence_reshaped = features_sequence[:, np.newaxis, np.newaxis, :]
        # Repeat the features to match the spatial dimensions of the runner_sequence
        features_repeated = np.repeat(np.repeat(features_sequence_reshaped, runner_sequence.shape[1], axis=1), runner_sequence.shape[2], axis=2)
    
        # The runner matrix we want to predict is for the next day, not included in the last 7 days sequence
        current_day = combined_data.iloc[i+7]
        next_day_runner_matrix.append(current_day['runner_matrix'])

        # Combine runner sequence with features sequence along the last axis
        combined_sequence = np.concatenate((runner_sequence, features_repeated), axis=-1)
        sequences.append(combined_sequence)

    # Convert lists to numpy arrays for the model
    X = np.array(sequences)
    y = np.stack(next_day_runner_matrix)

    # Convert X and y to a consistent data type (e.g., float32)
    X = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('float32')

    print("X shape:", X.shape)
    print("X data type:", X.dtype)
    print("y shape:", y.shape)
    print("y data type:", y.dtype)

    # Ensure all values are finite and not NaN
    assert np.isfinite(X).all(), "X contains non-finite values"
    assert np.isfinite(y).all(), "y contains non-finite values"

    y = y.reshape(y.shape[0], -1)  # Reshape y to be 2D: (52, 31*33)
    print("y shape:", y.shape)

    return X, y

def read_and_precess_dataset_default(datapath, feadturepath, targetfactor, predictor):
    X_df = pd.read_csv(datapath)
    X_df['datetime'] = pd.to_datetime(X_df['date'])
    # shift the data by one day
    X_df['datetime'] = X_df['datetime'] - pd.DateOffset(days=1)
    grid_height = 31
    grid_width = 33

    grouped_by_date = X_df.groupby("date")

    reshaped_data = []

    for date, group in grouped_by_date:

        reshaped_grid = group[targetfactor].to_numpy().reshape(grid_height, grid_width, -1)
        # flattened_grid = reshaped_grid.flatten()  # 展平为一维数组

        # 将数据添加到列表中
        reshaped_data.append((date, reshaped_grid))

        # reshaped_data.append(group["keep_length"].to_numpy().reshape(grid_height, grid_width))

    reshaped_array = np.array([x[1] for x in reshaped_data])

    # 输入数据集
    input_data = []
    # 目标数据集
    target_values = []

    # 这里我们创建了基于时间步的输入和目标数据集
    # 使用循环来遍历数据，确保输入的长度是6，并且目标值是接下来的1个时间步

    for i in range(len(reshaped_array) - 7):
        # 输入数据是连续的6个时间步
        input_data.append(reshaped_array[i:i+7, :])

        # 目标值是紧接在这6个时间步后的那个时间步
        target_values.append(reshaped_array[i+7, :])

    # 将输入数据和目标值转换为numpy数组
    input_data = np.array(input_data)
    target_values = np.array(target_values)
    target_values = target_values.reshape(-1, 31*33)

    return input_data, target_values

def model_and_train_default(X, y, grid_height, grid_width):

    # Model structure
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu'), input_shape=(7, 31, 33, 1)),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.Flatten()),  # Flattening 2D features
        layers.LSTM(50, activation='relu', return_sequences=False), # LSTM layer
        layers.Dense(31*33, activation='linear')  # The output layer 
    ])

    # # Define the model 
    # model = keras.Sequential([ 
    #     layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu'), input_shape=(6, 31, 33, 1 + additional_features.shape[1])), 
    #     layers.TimeDistributed(layers.MaxPooling2D((2, 2))), 
    #     layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')), 
    #     layers.TimeDistributed(layers.Flatten()), 
    #     layers.LSTM(50, return_sequences=True), 
    #     layers.Dense(31*33, activation='linear') # Adjust according to your output requirements 
    #     ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='mse',  # 使用均方误差
        metrics=['mae']  # 平均绝对误差
    )

    # 训练模型
    history = model.fit(
        X,  # 输入数据
        y,  # 目标值
        epochs=1600,  # 训练轮数
        batch_size=8,  # 批量大小
        validation_split=0.3  # 20% 用于验证
    )

    return model, history


def model_and_train_weather(X, y, grid_height, grid_width):
    # Define the model
    model = Sequential()
    # TimeDistributed CNN layers for spatial features
    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'), input_shape=(7, grid_height, grid_width, X.shape[-1])))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    # LSTM layer for temporal features
    model.add(LSTM(units=50, return_sequences=False))
    # Output layer to predict the runner matrix
    model.add(Dense(units=grid_height * grid_width, activation='linear'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model - Adjust epochs and batch_size based on your computational capacity and dataset size
    history = model.fit(X, y, epochs=1600, batch_size=32, validation_split=0.2)
    

    return model, history

def viz(y, output_array, now, targetfactor, predictor):
    grid = gpd.read_file('../03_input/grid.geojson')    
    # match the output array with the grid
    grid['original_runner_matrix'] = y[0].flatten()
    grid['predicted_runner_matrix'] = output_array[0].flatten()

    # subplot, left is the original runner matrix(keep_num), right is the predicted runner matrix(predicted_runner_matrix)
    # use the same color scale
    max_value = max(grid['original_runner_matrix'].max(), grid['predicted_runner_matrix'].max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    grid.plot(column='original_runner_matrix', ax=ax1, legend=True,  vmin=0, vmax=max_value)
    ax1.set_title('Original Runner Matrix')
    ax1.axis('off')
    grid.plot(column='predicted_runner_matrix', ax=ax2, legend=True,  vmin=0, vmax=max_value)
    ax2.set_title('Predicted Runner Matrix')
    ax2.axis('off')
    # plt.show()
    plt.savefig(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__output.png')

# def main1():
#     # get the current date and time
#     import datetime
#     now = datetime.datetime.now()
#     now = now.strftime("%Y_%m_%d_%H_%M_%S")

#     datapath = '../03_input/combined_running_data.csv'
#     feadturepath = '../03_input/feature.csv'
#     targetfactor = "keep_num"
#     predictor = "weather"
#     grid_height = 31
#     grid_width = 33

#     X, y = read_and_precess_dataset_default(datapath, feadturepath, targetfactor, predictor)
#     model = model_and_train_default(X, y, grid_height, grid_width)

#     # save the model
#     model.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__model.h5')

#     output_array = model.predict(X)
#     output_array = output_array.reshape(output_array.shape[0], grid_height, grid_width)

#     # save the y to json
#     np.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__y.npy', y)
#     # save the output to json
#     np.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__output.npy', output_array)
    
#     # viz
#     viz(y, output_array, now, targetfactor, predictor)

def main(predictor, targetfactor):
    # get the current date and time
    import datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")

    datapath = '../03_input/combined_running_data.csv'
    feadturepath = '../03_input/feature.csv'
    grid_height = 31
    grid_width = 33

    if predictor == 'weather':
        X, y = read_and_precess_dataset_default(datapath, feadturepath, targetfactor, predictor)
        model, history = model_and_train_default(X, y, grid_height, grid_width)

    if predictor == 'default':
        X, y = read_and_precess_dataset_default(datapath, feadturepath, targetfactor, predictor)
        model, history = model_and_train_default(X, y, grid_height, grid_width)
        y = y.reshape(y.shape[0], grid_height*grid_width)
    # save the model
    model.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__model.h5')

    # save the history.history # history.history.keys() # dict_keys(['loss', 'val_loss'])
    np.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__history.npy', history.history)

    output_array = model.predict(X)
    output_array = output_array.reshape(output_array.shape[0], grid_height, grid_width)

    # save the y to json
    np.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__y.npy', y)
    # save the output to json
    np.save(f'../04_output/runner_matrix__{now}__{targetfactor}__{predictor}__output.npy', output_array)
    
    # viz
    viz(y, output_array, now, targetfactor, predictor)


if __name__ == '__main__':

    import argparse

    # 创建一个解析器
    parser = argparse.ArgumentParser(description='这是一个预测器程序')

    # 添加命令行参数
    # --mode 参数，用于指定模式
    parser.add_argument('--mode', type=str, required=True, help='模式选择')
    # --var 参数，用于指定变量
    parser.add_argument('--var', type=str, required=True, help='变量名称')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用传入的参数
    print(f'模式是: {args.mode}')
    print(f'变量是: {args.var}')

    predictor = args.mode

    targetfactor = args.var

    if predictor in ['weather','default'] and targetfactor in ['keep_num','length_average','keep_length']:
        main(predictor, targetfactor)
    else:
        print('the mode is not supported')