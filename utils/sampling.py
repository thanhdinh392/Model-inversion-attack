import numpy as np

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):  # num_users = 5 hoặc 10
    dict_users = {}
    num_classes = 10  # MNIST có 10 class
    class_shards = num_users * 2  # Mỗi client sẽ nhận dữ liệu từ 2 class khác nhau
    num_imgs = len(dataset)  # Tổng số lượng hình ảnh trong dataset
    labels = dataset.train_labels.numpy()  # Lấy nhãn của dataset MNIST

    # Chia các ảnh thành từng class
    dict_class_idxs = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Shuffle chỉ mục để đảm bảo phân phối ngẫu nhiên trong mỗi class
    for i in dict_class_idxs:
        np.random.shuffle(dict_class_idxs[i])

    # Gán dữ liệu non-IID: mỗi client nhận 2 class khác nhau
    class_per_client = num_classes // (num_users // 2)  # Mỗi client có 2 class
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Tạo tập hợp chỉ số của 2 class cho mỗi client
    for i in range(num_users):
        chosen_classes = np.random.choice(range(num_classes), 2, replace=False)  # Lấy 2 class cho mỗi client
        num_imgs_client = np.random.randint(1000, 6000)  # Số ảnh của mỗi client có thể khác nhau

        for cls in chosen_classes:
            class_idxs = dict_class_idxs[cls][:num_imgs_client // 2]  # Lấy một phần ảnh từ class
            dict_users[i] = np.concatenate((dict_users[i], class_idxs), axis=0)
            dict_class_idxs[cls] = dict_class_idxs[cls][num_imgs_client // 2:]  # Cập nhật lại class index sau khi phân chia

    return dict_users
# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     dict_users = {}
#     num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return dict_users

