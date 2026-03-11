def calculate_aqi(pm):

    if pm <= 30:
        return 40
    elif pm <= 60:
        return 80
    elif pm <= 90:
        return 120
    elif pm <= 120:
        return 160
    else:
        return 200