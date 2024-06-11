def main():
    epochs = input("Enter number of epochs or just skip it: ")

    if epochs.isdigit() and int(epochs) > 0:
        from app.train_and_save import train_and_save
        train_and_save(int(epochs))

    else:
        from app.load_and_generate_text import generate_text, model
        print('generated text:\n', generate_text(model, start_string=input('prompt: ')))


if __name__ == '__main__':
    main()
