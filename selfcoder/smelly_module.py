def process_data(data, user_type, is_admin):
    if not data:
        print('No data to process.')
        return False
    if user_type == 'premium':
        if is_admin:
            print('Processing premium admin data.')
            return True
        print('Processing premium user data.')
        return True
    if user_type == 'guest':
        if len(data) > 10:
            print('Processing long guest data.')
            return True
        print('Processing short guest data.')
        return True
