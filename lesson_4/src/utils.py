import numpy as np

def prefilter_items(data):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    prices = data.groupby('item_id', sort=False)['sales_value'].sum() / np.clip(data.groupby('item_id', sort=False)['quantity'].sum(), a_min=1, a_max=None)
    prices = prices.reset_index(name='price')
    too_cheap = prices.loc[prices['price'] <= 1, 'item_id'].tolist()
    data = data.loc[~data['item_id'].isin(too_cheap)]
    # Уберем слишком дорогие товары
    too_expensive = prices.loc[prices['price'] >= 50, 'item_id'].tolist()
    data = data.loc[~data['item_id'].isin(too_expensive)]
    # ...
    return data
