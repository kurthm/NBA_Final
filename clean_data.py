if __name__ == '__main__':
        
    from nba_draft import *

    # Create draft data
    data = get_draft(2020, 2023) 

    # Add college information where applicable
    data = add_colleges(data)


    # Print resulting dataframe
    print(data)
