from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from env import TELEGRAM_BOT_TOKEN


def agent(message):
    return f"{message} 받았고 특정 처리를 하겠습니다."


async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    if update.message is None or update.message.text is None:
        return

    user_messgae = update.message.text

    result = agent(user_messgae)

    await update.message.reply_text(result)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT, handler))

app.run_polling()
