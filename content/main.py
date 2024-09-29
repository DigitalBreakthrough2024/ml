# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import ast


def load_and_process_data(file_path):
    """
    Загружает данные из CSV-файла, обрабатывает столбцы с эмбеддингами.

    Args:
      file_path: Путь к CSV-файлу.

    Returns:
      DataFrame с обработанными данными.
    """

    df_stat = pd.read_csv(file_path)

    try:
        df_stat['combined_embedding'][0] = ast.literal_eval(df_stat['combined_embedding'][0])
    except:
        pass

    try:
        df_stat['description_embedding'] = df_stat['description_embedding'].apply(lambda x: ast.literal_eval(x))
    except:
        pass
    try:
        df_stat['combined_embedding'] = df_stat['combined_embedding'].apply(lambda x: ast.literal_eval(x))
    except:
        pass
    try:
        df_stat['category_embedding'] = df_stat['category_embedding'].apply(lambda x: ast.literal_eval(x))
    except:
        pass

    return df_stat


def rank_videos(df, total_views_weight=0.4, week_views_weight=0.35, freshness_weight=0.25, top_n=10):
    # Нормализуем каждый критерий
    df['total_views_norm'] = df['v_year_views'] / df['v_year_views'].max()
    df['week_views_norm'] = df['v_long_views_7_days'] / df['v_long_views_7_days'].max()
    df['freshness_norm'] = 1 - (df['days_since_publication'] / df['days_since_publication'].max())

    # Рассчитываем итоговый рейтинг для каждой строки
    df['ranking_score'] = (df['total_views_norm'] * total_views_weight +
                           df['week_views_norm'] * week_views_weight +
                           df['freshness_norm'] * freshness_weight)

    # Сортируем по итоговому рейтингу
    df_sorted = df.sort_values(by='ranking_score', ascending=False)

    # Возвращаем топ-N наиболее релевантных видео
    return df_sorted.head(top_n)[['video_id', 'ranking_score']]


def update_interest_vector(user_vector, video_vector, feedback, learning_rate=0.1):
    """
    Обновление вектора интересов пользователя на основе обратной связи (лайк/дизлайк).

    :param user_vector: Вектор интересов пользователя (np.array)
    :param video_vector: Вектор видео (np.array)
    :param feedback: Обратная связь пользователя (1 для лайка, -1 для дизлайка)
    :param learning_rate: Скорость обучения (шаг градиентного обновления)
    :return: Обновленный вектор интересов пользователя
    """
    # Проверяем, что размерности векторов совпадают
    assert len(user_vector) == len(video_vector), "Векторы пользователя и видео должны быть одинаковой размерности"

    # Градиентное обновление вектора интересов пользователя
    updated_user_vector = user_vector + learning_rate * feedback * (video_vector - user_vector)

    # Нормализация вектора для предотвращения его роста
    if sum(updated_user_vector) >= len(updated_user_vector):
        updated_user_vector = updated_user_vector / np.linalg.norm(updated_user_vector)

    return updated_user_vector


def cosine_similarity(user_vector, video_vector):
    """
    Вычисляет косинусное сходство между вектором пользователя и вектором видео.

    :param user_vector: Вектор интересов пользователя
    :param video_vector: Вектор видео
    :return: Косинусное сходство
    """
    # Преобразуем векторы в numpy массивы
    user_vector = np.array(user_vector)
    video_vector = np.array(video_vector)

    # Вычисляем косинусное сходство
    dot_product = np.dot(user_vector, video_vector)
    norm_user = np.linalg.norm(user_vector)
    norm_video = np.linalg.norm(video_vector)

    if norm_user == 0 or norm_video == 0:
        return 0.0  # Избегаем деления на ноль

    cosine_similarity = dot_product / (norm_user * norm_video)
    return cosine_similarity


def find_closest_videos(user_vector, df, n=10):
    """
    Находит N видео, наиболее близких к вектору интересов пользователя,
    используя косинусное сходство.

    :param user_vector: Вектор интересов пользователя
    :param df: DataFrame с данными о видео, содержащий столбец 'combined_embedding'
    :param n: Количество видео для возврата
    :return: DataFrame с N наиболее близкими видео
    """
    df['cosine_similarity'] = df['combined_embedding'].apply(lambda x: cosine_similarity(user_vector, x))
    df_sorted = df.sort_values(by='cosine_similarity', ascending=False)
    return df_sorted.head(n)[['video_id', 'cosine_similarity']]


def main(user_vector, video_ids, feedback, df_stat):
    """
    Основная функция для ранжирования видео на основе комбинации косинусного сходства и других параметров,
    а также обновления вектора интересов пользователя на основе оценок.

    Параметры:
    user_vector (list): Вектор интересов пользователя.
    video_ids (list): Список идентификаторов видео.
    feedback (list): Оценки пользователя для видео (1 — лайк, 0 — дизлайк).
    df_stat (pd.DataFrame): Датафрейм с данными о видео.

    Возвращает:
    pd.DataFrame: Отсортированный по релевантности пул видео.
    list: Обновленный вектор интересов пользователя.
    """

    # Шаг 1: Ранжируем видео по просмотрам и свежести
    rank_videos(df_stat)

    # Шаг 2: Ранжирование по косинусному сходству с вектором интересов пользователя
    video_pool = find_closest_videos(user_vector, df_stat)

    # Шаг 3: Обновляем вектор интересов пользователя на основе оценок
    for video in recommended_videos:
        video_vector = df_stat.loc[df_stat['video_id'] == video, 'combined_embedding']
        video_vector = video_vector.to_list()[0]
        updated_user_vector = update_interest_vector(user_vector, video_vector, feedback_dict[video])
        if video not in video_pool:
            # Добавляем видео в пул для дальнейшего слияния выборок двух систем
            archived_videos.append(video)

    return video_pool, updated_user_vector


if __name__ == "__main__":
    # Загрузка и препроцессинг данных:
    file_path = 'video_stat_50k_emb.csv'
    df_stat = load_and_process_data(file_path)
    print(df_stat.head())

    top_10_videos = rank_videos(df_stat)
    print(top_10_videos)

    LEN_USER_VECTOR = len(df_stat['combined_embedding'][0])

    user_vector = np.array([0.5] * LEN_USER_VECTOR)

    recommended_videos = top_10_videos['video_id'].to_list()


    # Псевдослучайные оценки на подборку
    feedback_dict = {
        recommended_videos[0]: 1,
        recommended_videos[1]: 0,
        recommended_videos[2]: -1,
        recommended_videos[3]: 1,
        recommended_videos[4]: 0,
        recommended_videos[5]: 1,
        recommended_videos[6]: -1,
        recommended_videos[7]: 0,
        recommended_videos[8]: 1,
        recommended_videos[9]: 1
    }

    video_pool = []
    archived_videos = []

# Обновление вектора

for video in recommended_videos:
    video_vector = df_stat.loc[df_stat['video_id'] == video, 'combined_embedding']
    video_vector = video_vector.to_list()[0]
    updated_user_vector = update_interest_vector(user_vector, video_vector, feedback_dict[video])
    if video not in video_pool:
        archived_videos.append(video)

user_vector = updated_user_vector
archived_videos

closest_videos = find_closest_videos(user_vector, df_stat, n=10)