
class ExpressionHandler:

    MAPPING = {
        "bình_thường": "Ngồi yên ",
        "cảm_ơn": "Cảm ơn 😘",
        "xin_chào": "Xin chào 🙋‍",
        "không": "Không 🤚",
        "xin_loi": "Xin Lỗi",
        "ban_that_tuyet_voi": "Bạn thật tuyệt vời",
        "biet_on": "Tôi rất biết ơn",
        "toi_khoe": "Tôi khỏe "
    }

    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        return ExpressionHandler.MAPPING[self.current_message]
