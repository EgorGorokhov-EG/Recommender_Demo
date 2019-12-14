import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data to test the system
DATA = {'user_id': np.arange(3),
        'name': ['John', 'Paul', 'George'],
        'tags': ['horror, drama, scifi', 'drama, advent, space', 'advent, action, history, scifi']}

EVENTS = {'event_id': np.arange(10),
          'name': ['Хоррор', 'Space_Oddity', 'Historical', 'British_Invasion', 'Парусный спорт', 'Meet-Up', 'Cinema',
                   'Гастрономическая ярмарка',
                   'Эскурсия в рзвалины', 'Аквапарк'],
          'tags': ['ужасы, экшн, приключение, игра', 'космос, экшн, игра, квест',
                   'история, драма, приключение, встреча',
                   'квест, британия, экшн', 'природа, приключения, экшн, спорт', 'культура, саморазвитие, бизнес',
                   'культура, отдых, история, встреча', 'природа, еда, встреча, отдых', 'культура, экскурсия, история',
                   'отдых, спорт, экшн']}


class Recommender:

    def __init__(self):
        self.df_users = pd.DataFrame(DATA)
        self.df_events = pd.DataFrame(EVENTS)

    def get_top(self, tags: str, limit: int = 3):
        """
        Args:
            tags: tags of user to whom we want to recommend
            limit: number of recommendations
        Returns:
            top of the recommendations for user with given tags
        """

        cv = CountVectorizer()
        user_events_matrix = cv.fit_transform(
            self.df_events['tags'].append(pd.Series(tags), ignore_index=True)
        )
        cos_sim_user_events = cosine_similarity(user_events_matrix)

        similar_events = list(enumerate(cos_sim_user_events[-1]))
        similar_events = sorted(similar_events)

        sorted_similar_events = sorted(similar_events, key=lambda x: x[1], reverse=True)[1:]

        result = []
        for i in range(limit):
            index = sorted_similar_events[i][0]
            result.append(self.df_events['name'][index])

        return result


if __name__ == '__main__':
    res = Recommender().get_top('космос, приключение, квест, игра')
    print(*res, sep='\n')
