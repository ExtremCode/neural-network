import telebot
import random
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from telebot import types

intense = {
    'hello': {
        'examples': ['привет', 'здоровки', 'добрый день', 'прив', '/start'],
        'responses': ['Приветули', 'Привет', 'Здоровки', 'Привет, пацан']
    },
    'how-are-you': {
        'examples': ['как дела', 'как себя чувствуешь', 'ты в порядке', 'что нового' 'как день прошел'],
        'responses': ['Хорошо', 'Замечательно',
                      'Лучше чем у тебя']
    },
    'yes': {
        'examples': ['ты жива', 'мне сделать', 'пудинг лор', 'могу я тебя спросить', 'я справлюсь'],
        'responses': ['Разумеется', 'Конечно', 'Да-да-да', 'Непременно так', 'Балдеж']
    },
    'no': {
        'examples': ['ты издеваешься', 'не маловата для такого'],
        'responses': ['Нет, нет и еще раз нет', 'Ни в коем случае', 'Ответ отрицательный']
    },
    'weather': {
        'examples': ['какая погода?', "как там на улице?", 'одеться теплее?'],
        'responses': ['Погода шикарная', 'Самое то с тобой погулять', 'Прохладно, как всегда.']
    },
    'undefined': {
        'examples': ['горшочек не вари'],
        'responses': ['Я тебя не поняла.', 'Мне не понятны эти слова.',
                      'Ты слишком непонятно пишешь, напиши нормально']
    },
    'exit': {
        'examples': ['пока', 'до встречи', 'удачи', 'увидимся', 'счастливо', 'всего хорошего'],
        'responses': ['Пока-пока.', 'Надеюсь, скоро еще увидимся!', 'Покаки))', 'До скорой встречи!']
    },
    'game': {
        'examples': ['поиграем', 'сыграем в игру', 'играть', 'поиграть', 'давай сыграем'],
        'responses': ['Я уже думала ты не предложишь', 'Только если с тобой', 'Только чур я первая хожу']
    },
    'doing': {
        'examples': ['что ты делаешь', 'чем занимешься', 'что планируешь делать', 'какие планы', 'чем занята',
                     'что ты умеешь', 'какая твоя роль', 'чем планируешь заниматься', '/help'],
        'responses': ['Да так, ничего интересного', 'Как обычно буду ждать твоего сообщения',
                      'Пока не знаю']
    },
    'joke': {
        'examples': ['анекдот', 'расскажи прикол', 'знаешь анекдот', 'анек', 'шутка', 'шутку'],
        'responses': ['Купил студент шляпу, а она ему как раз!',
                      'Студент с искусственным сердцем сидит на паре. Препод говорит отключить все электронные устройства, студент: до связи',
                      'Что общего у ворон и шампуня? Они щиплют глазки!']
    },
    'how-old': {
        'examples': ['тебе лет', 'какой твой возраст', 'уже взрослая', 'сколько тебе лет'],
        'responses': ['Уже достаточно большая)', 'Такое не прилично спрашивать',
                      'Старше тебя', 'Не скажу, это секрет.']
    },
    'walk': {
        'examples': ['погуляем', 'выйдем погулять', 'не хочешь сходить куда-нибудь', 'сходим куда-нибудь'],
        'responses': ['С тобой хоть на край света', 'Куда пойдем?', 'Сейчас подожди, я из-за компа вылезу']
    },
    'what-you-love': {
        'examples': ['что тебе нравится','что ты любишь','чего ты хочешь'],
        'responses': ['Мне нравишься ты', 'Я люблю смотреть разные сериалы',
                      'Я бы хотела объездить весь мир',
                      'Я бы сказала, что люблю тебя, не будь ты таким...']
    },
    'your-name': {
        'examples': ['как тебя зовут', 'твое имя', 'как тебя там', 'как тебя назвают'],
        'responses': ['Имя мое Мандаринка', 'Меня зовут Мандаринка',
                      'Я Мандиринка по паспорту', 'Называй меня Мандаринка']
    },
    'your-high': {
        'examples': ['какой твой рост', 'ты выше меня', 'достаешь до поручня в автобусе'],
        'responses': ['Я не выше тебя', 'Я выше тебя только в социальном плане',
                      'Скажу так: меня не считают за ребенка']
    },
    'your-weight': {
        'examples': ['сколько ты весшиь', 'какой твой вес', 'ты тяжелая'],
        'responses': ['Такое я не скажу, прости', 'Это слишком личное, чтобы говорить тебе',
                      'Я легкая как пушинка']
    },
    'mood': {
        'examples': ['как твое настроение', 'что ты чувствушь', 'как твои чувства'],
        'responses': ['Я счастлива общаться с тобой',
                      'Я не чувтсвую печали, чтобы ты не смог меня обидеть',
                      'Сегодня настроение такое хорошее']
    },
    'studing': {
        'examples': ['где ты училась', 'какое у тебя образование', 'у тебя есть степень'],
        'responses': ['Я закончила Махачкалински ВШЭ', 'Я с красным дипломом закончила факультет виноделия ДГТУ',
                      'У меня есть диплом Синергии, лучше не шути со мной']
    },
    'home': {
        'examples': ['где твой дом', 'где ты живешь', 'где тебя встретить'],
        'responses': ['Я живу только в твоем сердце',
                      'У меня нет конкретного места, мой дом - вся планета',
                      'Я живу там, где в меня верят и где я нужна']
    },
    'opinion': {
        'examples': ['твое мнение по этому', 'что ты думаешь об этом', 'что ты думаешь', 'как считаешь'],
        'responses': ['Это слишком сложно для меня, спроси что-нибудь полегче',
                      'Я в этом вопросе не разбираюсь, прости. Ничего сказать не могу',
                      'Спросил же такое, конечно. Я не знаю']
    },
    'more': {
        'examples': ['еще', 'давай дальше', 'еще раз', 'повтори'],
        'responses': ['Конечно', 'Как скажешь', 'Разумеется', 'Без проблем']
    },
    'question': {
        'examples': ['хочу тебя спросить', 'можно задать вопрос'],
        'responses': ['Да, что такое?', 'О чем ты хочешь поговорить?', 'Что тебя беспокоит?']
    }
}

X = []
Y = []
for name in intense:
    for phrase in intense[name]['examples']:
        X.append(phrase)
        Y.append(name)
    for phrase in intense[name]['responses']:
        X.append(phrase)
        Y.append(name)
vectorizer = CountVectorizer()
vectorizer.fit(X)
X_vec = vectorizer.transform(X)  # vectorized

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, alpha=1e-4,
                      solver='adam', verbose=10, tol=1e-5, random_state=1,learning_rate_init=.1)
model.fit(X_vec, Y) # train model

def clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\n]', '', text)
    return text

def txt_match(user_msg, examp):
    user_msg = clean(user_msg)
    examp = clean(examp)
    if user_msg.find(examp) >= 0 or examp.find(user_msg) >= 0:
        return True
    return nltk.edit_distance(user_msg, examp) / len(examp) < 0.2

def get_intent(text):
    text_vec = vectorizer.transform([text])
    intent = model.predict(text_vec)[0]
    return intent

def get_response(intent):
    return random.choice(intense[intent]['responses'])

bot = telebot.TeleBot('your Token')

count_before = -1
@bot.message_handler(content_types=['text'])
def send_repl(message):
    global count_before # for change and use global parametr
    count_before += 1
    if count_before == 8:
        keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        keyboard.add(types.KeyboardButton(text='да'), types.KeyboardButton(text='нет'))
        bot.send_message(message.from_user.id,'Как я поняла, ты хочешь стать крутым специлистом в IT?',
                         reply_markup=keyboard)
    if message.text == 'да' and count_before == 10:
        keyboard = types.InlineKeyboardMarkup()
        keyboard.add(types.InlineKeyboardButton('тыкай сюда',
                                                url='https://vityaschel.github.io/YouWantToBecomeAProgrammer/'))
        bot.send_message(message.from_user.id, 'Я вижу, ты хочешь стать крутым прогером, тогда',
                         reply_markup=keyboard)
    else:
        print(count_before, message.text)
        intent = get_intent(message.text)
        prev_intent = ""
        if intent == 'game' or intent == 'joke':
            prev_intent = intent
        answ = get_response(intent)
        bot.send_message(message.from_user.id, answ)  # send message for user
        count_before += 1
        if intent.find('game') != -1:
            game(message)
        elif intent.find('more') != -1:
            if prev_intent == 'game':
                print("there")
                game(message)
            elif prev_intent == 'joke':
                bot.send_message(message.from_user.id, random.choice(intense['joke']['responses']))

@bot.message_handler(content_types=["text"])
def game(message):
    number = random.randint(0, 5)
    keyboard = types.InlineKeyboardMarkup(row_width=5)
    item0 = types.InlineKeyboardButton(text="0", callback_data="0"+str(number))
    keyboard.add(item0)
    item1 = types.InlineKeyboardButton(text="1", callback_data="1"+str(number))
    keyboard.add(item1)
    item2 = types.InlineKeyboardButton(text="2", callback_data="2"+str(number))
    keyboard.add(item2)
    item3 = types.InlineKeyboardButton(text="3", callback_data="3"+str(number))
    keyboard.add(item3)
    item4 = types.InlineKeyboardButton(text="4", callback_data="4"+str(number))
    keyboard.add(item4)
    item5 = types.InlineKeyboardButton(text="5", callback_data="5"+str(number))
    keyboard.add(item5)
    item6 = types.InlineKeyboardButton(text="6",
                                       url='https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley')
    keyboard.add(item6)
    bot.send_message(message.from_user.id, 'Я загадала число от 0 до 5. Попробуй его угадать!',
                     reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    if int(call.data[0]) < int(call.data[1]):
        bot.send_message(call.from_user.id, 'Мое число больше, подумай еще.')
    elif int(call.data[0]) > int(call.data[1]):
        bot.send_message(call.from_user.id, 'Я загадала не настолько большое число')
    else:
        bot.send_message(call.from_user.id, 'Отлично! Ты справился, поздравляю!')

bot.polling(none_stop=True, interval=0)
