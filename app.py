import telebot
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
from telebot import types
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn.functional as F

# Initialize the Telegram bot
bot = telebot.TeleBot('7462078476:AAEwek8t4YmbUaUYGfMUWPqJwSW8DxBqaso')

# Define the number of classes in your model checkpoint
num_classes = 4  # Adjust according to your model

# Load the Vision Transformer model from the Hugging Face model hub with custom num_labels and ignore_mismatched_sizes=True
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load the state dictionary from the saved model, removing 'module.' prefix if it exists
state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v

# Load the modified state dictionary into the model
model.load_state_dict(new_state_dict)
model.eval()

# Define language constants
UZBEK = 'uz'
ENGLISH = 'en'
RUSSIAN = 'ru'

# Define welcome messages
WELCOME_MESSAGES = {
    UZBEK: '''- Assalomu alaykum, Men dasturchi Bahodir Alayorov tomonidan tarbiyalangan sun'iy intellektman.
    - Bemorning miyasining rasmiga qarab, bemorda o'simta turi bor yoki yo'q ekanligini aniqlab beraman.
    - Mening aniqliligim 99.31 %. Men tarbiyalangan rasmlar, malakalik shifokorlar tomonidan ko'rib chiqilgan va tasdiqlanganman.
    - Muhim: Men test rejimida ishlamoqdaman, iltimos, menga ishonib hulosa qilmang, malakalik shifokorga murojat qiling.
    Boshlash uchun, iltimos bemor miyasining rasmini yuboring. Muhim: rasm ko'rinishida, file emas.''',

    ENGLISH: '''- Hello, I am a trained artificial intelligence by developer Bahodir Alayorov.
    - Looking at the picture of the patient's brain, I can determine whether the patient has a tumor or not.
    - My model accuracy is 98 %. I was trained with pictures, carefully reviewed and approved by professional doctors.
    - Important: I am working in test mode, please do not have a conclusion based on my response, consult a qualified doctor.
    To begin, please send the MRI image of the patient's brain. Note: as a picture, not a file.''',

    RUSSIAN: '''- Здравствуйте, я обученный искусственный интеллект от разработчика Bahodir Alayorov.
    - Глядя на снимок мозга больного, я могу определить, есть ли у больного опухоль или нет.
    - У меня действительно высокая точность (98 %). Меня обучали фотографиям, тщательно просмотренным и одобренным профессиональными врачами.
    - Важно: я работаю в тестовом режиме, пожалуйста, не делайте вывод по моему ответу, обратитесь к квалифицированному врачу.
    Для начала отправьте МРТ снимок мозга пациента. Примечание: как изображение, а не файл.''',
}

user_data = {}

# Handler function for the language selection buttons
@bot.message_handler(commands=['start'])
def send_welcome(message):
    # Create the keyboard markup with the language selection buttons
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    markup.add(
        types.KeyboardButton(text='🇺🇿 O\'zbek'),
        types.KeyboardButton(text='🇬🇧 English'),
        types.KeyboardButton(text='🇷🇺 Русский')
    )

    # Send the welcome message with the language selection buttons
    msg = bot.send_message(message.chat.id, 'Tilni tanlang / Please select your language / Выберите язык', reply_markup=markup)

    # Store the selected language in the user data for future use
    bot.register_next_step_handler(msg, process_language_selection)

# Handler function for processing the language selection
def process_language_selection(message):
    try:
        # Get the selected language from the button text
        flag_emoji, language_name = message.text.split()
        selected_language = {
            '🇺🇿': UZBEK,
            '🇬🇧': ENGLISH,
            '🇷🇺': RUSSIAN
        }[flag_emoji]

        # Store the selected language in the user data for future use
        user_data[message.chat.id] = {'language': selected_language}

        # Send the welcome message in the selected language
        welcome_message = WELCOME_MESSAGES[selected_language]
        markup = types.ReplyKeyboardRemove(selective=False)
        bot.send_message(message.chat.id, welcome_message, reply_markup=markup)
    except:
        # If an error occurs, default to English and instruct to use /start command to change language
        selected_language = ENGLISH  # Default to English
        user_data[message.chat.id] = {'language': selected_language}

        # If an error occurs, ask the user to select a language
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
        markup.add(
            types.KeyboardButton(text='🇺🇿 O\'zbek'),
            types.KeyboardButton(text='🇬🇧 English'),
            types.KeyboardButton(text='🇷🇺 Русский')
        )
        msg = bot.send_message(message.chat.id, 'Tilni tanlang / Please select your language / Выберите язык', reply_markup=markup)
        bot.register_next_step_handler(msg, process_language_selection)

        # Instruct user to use /start command to change language
        bot.send_message(message.chat.id, "You can change the language by using the /start command. /start tugmasi orqali tilni o'zgartira olasiz.")

# Define the transformation for image preprocessing
def transform_image(image):
    return feature_extractor(images=image, return_tensors="pt")["pixel_values"]

# Define the message handler for processing images
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        if message.chat.id not in user_data or 'language' not in user_data[message.chat.id]:
            # If language is undefined, default to English
            selected_language = ENGLISH
        else:
            selected_language = user_data[message.chat.id]['language']

        # Get the image file ID
        file_id = message.photo[-1].file_id
        # Download the image
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        # Convert the image to PIL format
        img = Image.open(BytesIO(downloaded_file))
        if img is None:
            bot.reply_to(message, "Error: Unable to open the image.")
            return
        # Handle different image formats
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Preprocess the image
        img_tensor = transform_image(img)

        # Make predictions
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.logits, 1)

        # Define the classes
        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitar']
        result = classes[predicted.item()]
        # Inside handle_image function after model prediction
       # Inside handle_image function after model prediction
        print(f"Raw logits: {outputs.logits}")
        print(f"Predicted class index: {predicted.item()}")

        # Softmax activation example
        

        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitar']
        result = classes[predicted.item()]


        # Adjusting class mapping based on model output index
        

        CLASS_NAMES = {
            UZBEK: {
                'Class 1': f"Glioma: {result}",
                'Class 2': f"Meningioma: {result}",
                'Class 3': f"No Tumor: {result}",
                'Class 4': f"Pituitar: {result}",
            },
            ENGLISH: {
                'Class 1': f"Glioma: {result}",
                'Class 2': f"Meningioma: {result}",
                'Class 3': f"No Tumor: {result}",
                'Class 4': f"Pituitar: {result}",
            },
            RUSSIAN: {
                'Class 1': f"Глиома: {result}",
                'Class 2': f"Менингиома: {result}",
                'Class 3': f"Нет опухоли: {result}",
                'Class 4': f"Питуитар: {result}",
            },
        }

        

        # Send the prediction result
        class_map = {
            0: 'Glioma',
            1: 'Meningioma',
            2: 'No Tumor',
            3: 'Pituitar',
        }

        # Example thresholding based on softmax probabilities
        probabilities = F.softmax(outputs.logits, dim=1)
        confidence_threshold = 0.5  # Adjust as needed
        predicted_index = torch.argmax(probabilities, dim=1).item()

        

        # Send the prediction result
        bot.send_chat_action(message.chat.id, 'typing')
        bot.send_message(message.chat.id, CLASS_NAMES[selected_language].get(result, result))

    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")

# Polling the bot to receive messages
bot.polling()
