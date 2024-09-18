# %pip install fastparquet
# %pip install pandas pyarrow fastparquet

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


#Warning surpressors for slice work on the df

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)




def build_unified_view() -> pd.DataFrame:

    # Ignore the following line. You will redefine unified_view below.
    unified_view = pd.DataFrame()

    production_data_df = pd.read_parquet("production_logging_data.parquet")
    # BLOCK 1 - PROCESSING OF PRODUCTION DATA CODE BELOW (prepare unified_view base)
    # Hint: Since the part type only logs on change, the first task is to ensure
    # the product type reflects for each unique part produced. Additionally the
    # resultant dataframe should be dense.
    print(production_data_df.PART_TYPE.unique())

    #Check for the number of missing values and repeated rows in the data set prior to adjustment
    print('No. of missing values in the production data df: ',production_data_df.isna().sum())
    print('No. of repeated values in the production data df: ',production_data_df.duplicated().sum())
    

    # print(production_data_df[production_data_df['unique_part_identifer'].isna()])

    #We know that there are two rows in which the parts are missing UIDs and timestamps. 
    #We know there are two rows that contain the part_type of the components
    #From the top, it is shown that these rows contain missing values for 
    #UID and start times. 
    #We should remove these rows after adding the part type to the rest of rows.
    #Becasue they do not contain any useful information since no production is occuring
     
    #Take away the warnings for this particular part of the code. It can be done in a way
    #that uses .iloc and doesn't need warning suppressors. 


    #Add the missing PART_TYPE to each of the rows that it is missing in. 
    for i in range(len(production_data_df.PART_TYPE)):
        current_part = production_data_df.PART_TYPE[i]
        updated_part = current_part
        if current_part == None:
            updated_part = production_data_df.PART_TYPE[i-1]
        production_data_df.PART_TYPE[i]= updated_part


    #Now to drop the rows with nan values in them

    production_data_df.dropna(inplace=True)

    # ###################################

    pressure_data_df = pd.read_parquet("pressure_data.parquet")
    # BLOCK 2 - PROCESSING OF PRESSURE DATA AND MERGING INTO PRODUCTION DATA CODE BELOW
    # Hint: Extract the most important part of the pressure profile, as described in the readme file

    #Check for missing dates which could interfear with our code. 
    print('No. missing values in pressure df', pressure_data_df.isna().sum()) #No missing dates in the data set. 
    
    #Since there is no unique identifier, bar the index, we should use the index as the uid
    #to check for repeated rows in pressure 

    df_pres= pressure_data_df.reset_index()
    print('No. repeated rows in pressure df', df_pres.duplicated().sum()) 

    #So we can see there are no missing or repeated rows in this dataset. 


    #Define a function to extract the maximum pressure and time elapsed 
    #on a subgroup of a df.

    def max_pressure_time_ellapsed(group):
        max_pressure = group['pressure_sensor'].max()

        time_of_mp = group[group['pressure_sensor']==max_pressure].iloc[0].name
        # print('time of mp:', time_of_mp)
        start_time= group.iloc[0].name
        # print('start time:',start_time)
        
        time_elapsed = (time_of_mp - start_time).total_seconds()/60
        # print(time_elapsed)

        return(pd.Series({'max_pressure_reached': max_pressure, 'time_to_max_pressure(minutes)': time_elapsed}))




    #Resample the df to break it into 30 minute intervals for feature extraction.
    df_pressure_max= pressure_data_df.resample('30T').apply(max_pressure_time_ellapsed)



    #The above function calaulates the highest temperature reached every 30 minutes
    #It contains within it the cycle start time, the max pressure of the cycle and
    #The time it takes to reach said pressure. To merge, we can merge on the cycle 
    #Start time or on the index that was returned in the process. 

    unified_view = pd.merge(production_data_df, df_pressure_max, left_index= True, right_index=True)


    # ###################################

    temperature_data_df = pd.read_parquet("casting_temperature_data.parquet")
    # BLOCK 3 - PROCESSING OF PRESSURE DATA AND MERGING INTO PRODUCTION DATA CODE BELOW
    # Hint: You can use pandas merge_asof function for this

    temp_df= temperature_data_df.reset_index()

    #Check for any missing values in the temp dataset and repeated rows.

    print('Num. missing temp data vals:', temperature_data_df.isna().sum())
    
    #As before, no UID means we need to consider the index as the UID

    print('Num duplicates in temp dataset: ', temp_df.duplicated().sum())

    #There are 5 missing values for the temperature of the casting. 
    #There are no duplicated rows. 

    #Let us remove the missing casting temperatures from this dataset. 

    temp_df.dropna(inplace=True)
    


    unified_view=pd.merge_asof(unified_view,temp_df, left_on= 'cycle_start_timestamp',right_on='index',direction= 'forward') 


    # ###################################

    silicon_data_df = pd.read_parquet("furnace_silicon_data.parquet")
    # BLOCK 4 - PROCESSING OF PRESSURE DATA AND MERGING INTO PRODUCTION DATA CODE BELOW
    # Hint: You can use pandas merge_asof function for this

    #The silicon levels will be assumed as stable for the remaining four hours.
    #So we will assume the levels don't change between their measurements. 
    # print(silicon_data_df.head())
    silicon_df= silicon_data_df.reset_index()

    #Again, check for missing values and duplicated rows
    print('Num. missing silicon data vals:', silicon_df.isna().sum())
    print('Num duplicates in silicon dataset: ', silicon_df.duplicated().sum())


    unified_view=pd.merge_asof(unified_view,silicon_df, left_on= 'cycle_start_timestamp',right_on='index',direction= 'backward') 
    
    #rename the indeces of from the merges. We will remove these at the end. 
    unified_view.rename(columns={'index_x':'Time_of_Temp_measure', 'index_y': 'Time_of_silicone_measure'}, inplace=True)
    

    # ###################################

    # BLOCK 5 - CRITICAL REVIEW OF THE FINAL UNIFIED VIEW (ADD CODE AND/OR COMMENTS + IMAGES)
    #Things to consider for the critial review- 
    
    #######Consider the dataset as a whole
    print('Number of nan values in the unified view dataset: ', unified_view.isna().sum())
    #We now have a dense dataset that contains no missing values. 



    ########variations of max_pressures.

    print('Number of repeated UIDs:',len(unified_view)-unified_view.unique_part_identifer.nunique())
    #So there are no repetitions of the unique parts, which can make analysis challenging 
    #This is because th size of the parts can influes a number of behaviours of the sytem. 

    #That being said, Let's look at the descriptive statistics of the dataset. 
    #In particular, let's look at the time to max pressure and the max pressures reached

    #Please uncomment for descriptive statistics.

    # print(unified_view[['max_pressure_reached', 'time_to_max_pressure(minutes)']].describe())


    #By the descriptive statistics, we can see that there is a value of 0 in the maximum pressure
    #And it occurred with 0 minutes. Upon inspection, it was deduced that this value occured
    #Due to the data being incomplete for this particular UID. For this reason, it will be removed
    #From the data set. 

    unified_view= unified_view.drop(unified_view.index[-1])
    print('Max Pressure and Time to max pressure descriptive statistics:' )
    print(unified_view[['max_pressure_reached', 'time_to_max_pressure(minutes)']].describe())

    #From the descriptive statistics, we can see that there is a small standard deviation
    #In the maximum pressure reached by the system. The same goes for the time it takes
    #For the sytem to pressurize. Let us consider the plot of the pressure vs time graph

    plt.scatter(unified_view['time_to_max_pressure(minutes)'], unified_view['max_pressure_reached'])

    plt.xlabel('Time to Max Pressure (minutes)')
    plt.ylabel('Max Pressure Reached')
    plt.title('Max Pressure vs. Time to Max Pressure')

    plt.show()

    #We notice from the graph entitled Max Pressure vs Time to max pressure that the system
    #Pressurised to the same pressure over varying times. This can be due to a number of factors
    #Atmospheric conditions, component size and other possible contributing factors. 
    #This could also be an indication that the mechanism requires maintenance,
    #But more data is required in order to predict this measure. 


    ########variations of temperatures for UID's.

    #Let us now consider the descriptive statistics of the temperatures of the castings
    #For the UIDs:

    print('Casting Temperature descriptive statistics:' )
    print(unified_view['casting Temperature'].describe())

    #In these descriptive statistics, it is again shown that there is little standard deviation
    #In casting temperature with a maximum difference being around 6 degrees. 
    #Although small, the difference on fluid viscosity in this range might 
    #Change, this can lead to inconsistent results.  
    #Manufacturing might benefit from optimising the casting temperature to produce more
    #consistent results. 

    ########Consider the amount of silicon used per each of the UID's

    #Silicon level are measured far less frequently than the other values. Let us consider 
    #There descriptive statistics. 
    print('Silicon content descriptive statistics:' )

    print(unified_view['furnace_silicon_content'].describe())
    # The standard deviation is low, but the difference between the maximum and the minumum
    #Amount of silicon is 0.6g, which is relatively high considereing the small amount of 
    #Silicon being used on average. 


    #Let us consider the plots of both silicon vs time to max pressure. 

    sns.boxplot(x='furnace_silicon_content', y='time_to_max_pressure(minutes)', data=unified_view)
    plt.xlabel('Furnace Slicon Content')
    plt.ylabel('Time to Max Pressure (minutes)')
    plt.title('Furnace Silicon Content vs. Time to Max Pressure')

    plt.show()

    #From the above plot, we notice that the amount of silicon in the furnace does
    #not significantly effect the time taken to reach the maximum pressure.
   
    #Let us consider the amount of silicon vs the temperature of the casting. 

    sns.boxplot(x='furnace_silicon_content', y='casting Temperature', data=unified_view)

    plt.xlabel('Furnace Slicon Content')
    plt.ylabel('Casting Temperature')
    plt.title('Casting Temperature vs Furnace Silicon Content')

    plt.show()

    #Silicon levels may affect the quality of the cast- affecting viscosity,
    #Hardness and oxidation prevention. These factors should be considered when 
    #Considering both temperature and silicon level in casting. 
    #Since temperature variations in casting affect similar aspects, these should
    #be considered together. 


    #Consider the total time it takes to create each of the components. 
    component= unified_view.PART_TYPE.unique()

    

    time_per_component=[]
    for i in component:
        comp_df = unified_view[unified_view['PART_TYPE']==i].reset_index()
        start = comp_df['cycle_start_timestamp'].iloc[0]
        end = comp_df['cycle_start_timestamp'].iloc[-1]
        time_per_component.append(end-start)
        print('Component', i, 'takes', end-start)

    #Here we see that BMW takes almost as long to produce a big wheel 
    #As Toyota takes to produce an entire car. 

    #######Final considerations

    #Despite the standard deviations being low in the cases displayed above, it 
    #Stands to reason that the process of casting can still be optimised. 
    #By standardising temperature, pressure and silicon levels, the casting process 
    #Should produce more consistent results.
    #Assuming the conditions of Laboratory are consistent.
    
    # ###################################


    unified_view=unified_view.drop(columns=['Time_of_Temp_measure','Time_of_silicone_measure'])
    print('Final unified view df:')
    print(unified_view.head())


    unified_view.to_csv( 'unified_view.csv')
    return unified_view


if __name__=='__main__':
    build_unified_view()