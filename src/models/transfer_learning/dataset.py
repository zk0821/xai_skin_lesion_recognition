from ham10000_dataset import HAM10000Dataframe


def main():
    ham_df_obj = HAM10000Dataframe()
    print(ham_df_obj.print_diagnosis_counts())


if __name__ == "__main__":
    main()
