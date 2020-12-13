# -*- coding: utf-8 -*-



import utils
import config


def main():
    """
        主函数
    """
    # 加载数据集
    word_index, x_train, x_test, y_train, y_test = utils.load_data()

    # 建立模型
    lstm_model = utils.build_model()

    if not config.load_model:
        # 训练模型
        utils.train_model(lstm_model, x_train, y_train)

    # 测试模型
    utils.do_prediction(lstm_model, x_test, y_test)


if __name__ == '__main__':
    main()
