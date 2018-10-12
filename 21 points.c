#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<string.h>
struct CARDS
{
	char image[52][8];
	char poker[52];
};
void put(void);
void judgement(int a1, int a2);
char correct(char l[], char a, int p);
void poker_print1(int n, int m[], int p, char pok[]);
void poker_print2(int n, int m[], int p, char pok[]);
int main(void) 
{   int i, index, primary, s, amo1, amo2, A1, A2, num, memory[23], quit, pick1, pick2;
    char ch, alt, letter[40];
    char image[52][8] = 
	{{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},
	{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},
	{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},
	{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"},
	{"Heart\0\0\0"},{"Diamond\0"},{"Club\0\0\0\0"},{"Spade\0\0\0"}};
    char poker[52] = {'A', 'A', 'A', 'A', 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 'J', 'J', 'J', 'J', 'Q', 'Q', 'Q', 'Q', 'K', 'K', 'K', 'K'};
    srand((unsigned)time(0));
    put();
    printf("Welcome!\n");
    quit = 0;
    primary = amo1 = amo2 = A1 = A2 = 0;
    while(quit == 0)
    {
    	put();
        printf("21POINTS GAME\n");
		put();
		printf("Enter a new line to start.\n");
		while(getchar() != '\n')
		continue;
		put();
		printf("Get your first card:\n""Participator          Player        Banker\n""        Card");
		pick1 = rand()%51+0;
		do
		{
			pick2 = rand()%51+0;
		}
		while(pick2 == pick1);

		if(pick1 > 3 && pick1 < 36)
	    {
	    	printf("%14s%2d", image[pick1], (int)poker[pick1]);
		    amo1 += (int)poker[pick1];
	    }
		else if(pick1 < 4)
		{   
		    printf("%14s A", image[pick1]);
			amo1 += 1;
			A1++;
		}
		else if(pick1 > 39)
		{
		   	printf("%14s %c", image[pick1], poker[pick1]);
		    amo1 += 10;
		}
		else
		{
		    printf("%13s 10", image[pick1]);
			amo1 += 10;
	
		} 
		if(pick2 > 3 && pick2 < 36)
	    {
	    	printf("%12s%2d", image[pick2], (int)poker[pick2]);
		    amo2 += (int)poker[pick2];
	    }
		else if(pick2 < 4)
		{   
		    printf("%12s A", image[pick2]);
			amo2 += 1;
			A2++;
		}
		else if(pick2 > 39)
		{
		   	printf("%12s %c", image[pick2], poker[pick2]);
		    amo2 += 10;
		}
		else
		{
		    printf("%11s 10", image[pick1]);
			amo2 += 10;
	
		} 
	    printf("\n\nPlayer's turn!\nEnter a new line to draw.\n");
        while(getchar() != '\n')
		continue;
	    i = 1;
	    alt = '1';
	    for(primary = 0; primary < 12; primary++)
	    memory[primary] = 0;
	    for(primary = 0; primary < 40; primary++)
	    letter[primary] = 0;
	    memory[0] = 100;
        while(alt == '1' && i < 12)
        {          
    	    index = rand()%51+0;
            memory[i] = index;
		    num = i;
    	    for(i--, s = 0; i > 0; i--, s++)
    	    {
    	        if(index == memory[i] || index == pick1 || index == pick2)
    		    {
    		   	    index = rand()%51+0;
    			    i += s;
    		   	    s = 0;
			    }
		    }
		    i = num + 1;
		    if(index > 3 && index < 40)
	        {
	    	    printf("Now you get: %s %d", image[index], (int)poker[index]);
		        amo1 += (int)poker[index];
		        printf("\nYou have: ");
                poker_print1(num, memory, pick1, poker);
                printf("\n");
			    put();
	        }
		if(index < 4)
		{   printf("Now you get: %s A", image[index]);
			amo1 += 1;
			A1++;
			printf("\nYou have: ");
		    poker_print1(num, memory, pick1, poker);
		    printf("\n");
			put();
		}
		if(index > 39)
		{
		   	printf("Now you get: %s %c", image[index], poker[index]);
		    amo1 += 10;
			printf("\nYou have: ");
		    poker_print1(num, memory, pick1, poker);
		    printf("\n");
			put();
		}
		if(amo1 > 21)
		{
		    printf("Player BUST!\nEnter to see amount.");
		    break;
		}
		if(i < 12)
		{
		    printf("\nDraw a card? (1 to draw, 0 to stop)\n");
		    alt = correct(letter, alt, primary);
		}
	}
	for(; A1 > 0; A1--)
	{
		if(amo1 + 10 < 22)
		amo1 += 10;
	}
	for(primary = 0; primary < 40; primary++)
	letter[primary] = 0;
	if(amo1 < 22)
	{
		printf("\nBanker's turn!\n");
		printf("Player have: ");
		--i;
		poker_print1(i, memory, pick1, poker);
   	    printf("\nEnter a new line to draw.\n");
	}
    while(getchar() != '\n')
	continue;
	i = 12;
	alt = '1';
    while(alt == '1' && i < 23)
    {   if(amo1 > 21)
		{
		    break;
		}       
    	index = rand()%51+0;
        memory[i] = index;
		num = i;
    	for(i--, s = 0; i > 0; i--, s++)
    	    {
    	        if(index == memory[i] || index == pick1 || index == pick2)
    		    {
    		    	index = rand()%51+0;
    			    i += s;
    		    	s = 0;
			    }
			}
		i = num + 1;
		if(index > 3 && index < 40)
	    {
	    	printf("Now you get: %s %d", image[index], (int)poker[index]);
		    amo2 += (int)poker[index];
		    printf("\nYou have: ");
		    poker_print2(num, memory, pick2, poker);
		    printf("\n");
			put();
	    }
		if(index < 4)
		{   printf("Now you get: %s A", image[index]);
			amo2 += 1;
			A2++;
			printf("\nYou have: ");
		    poker_print2(num, memory, pick2, poker);
		    printf("\n");
			put();
		}
		if(index > 39)
		{
		   	printf("Now you get: %s %c", image[index], poker[index]);
		    amo2 += 10;
			printf("\nYou have: ");
		    poker_print2(num, memory, pick2, poker);
		    printf("\n");
			put();
		}
		if(amo2 > 21)
		{
		    printf("Banker BUST!\nEnter to see amount.");
			while(getchar() != '\n')
		    continue;
		    break;
		}
		if(i < 23)
		{
		    printf("\nDraw a card? (1 to draw, 0 to stop)\n");
		    alt = correct(letter, alt, primary);
		} 
	}
	for(; A2 > 0; A2--)
	{
		if(amo2 + 10 < 22)
		amo2 += 10;
	}
	printf("\n\n");
    judgement(amo1, amo2);
	printf("\n        Player    Banker\n");
	printf("AMOUNT%8d%10d\n\n", amo1, amo2);
	amo1 = amo2 = A1 = A2 = 0;
	printf("Enter q to quit, enter a new line to continue: ");
    if((ch = getchar()) == 'q')
    {
	    put();
		printf("Goodbye!");
	    quit = 1;
	}    
    }
    
    return 0;
}
void put(void)
{
    printf("***********************************************************************************************************************\n");
} 
void judgement(int a1, int a2)
{
	if(a1 > 21)
	{
		if(a2 > 21)
		printf("Push.");
		else
		printf("Banker wins!");
	}
	else
	{
		if(a2 > 21)
		printf("Player wins!");
		else
		{
			if(a1 == a2)
			printf("Push.");
			else
			{
				if(a1 > a2)
				printf("Player wins!");
				else
				printf("Banker wins!");
			}
		}
	}
}
void poker_print1(int n, int m[], int p, char pok[])
{
	for(; n > 0; n--)
		        {
		        	if(m[n] < 4)
		    	    printf("A ");
		    	    if(m[n] > 3 && m[n] < 40)
				    printf("%d ", (int)pok[m[n]]);
				    if(m[n] > 39)
				    printf("%c ", pok[m[n]]);
			    }
		        if(p > 3 && p < 40)
	    	    printf("%d ", (int)pok[p]);
		        if(p < 4)
		        printf("A ");
		        if(p > 39)
		   	    printf("%c ", pok[p]);
}
void poker_print2(int n, int m[], int p, char pok[])
{
	for(; n > 11; n--)
		        {
		        	if(m[n] < 4)
		    	    printf("A ");
		    	    if(m[n] > 3 && m[n] < 40)
				    printf("%d ", (int)pok[m[n]]);
				    if(m[n] > 39)
				    printf("%c ", pok[m[n]]);
			    }
		        if(p > 3 && p < 40)
	    	    printf("%d ", (int)pok[p]);
		        if(p < 4)
		        printf("A ");
		        if(p > 39)
		   	    printf("%c ", pok[p]);
}
char correct(char l[], char a, int p)
{
	        gets(l);
		    a = l[0];
		    while(strlen(l) != 1 || (a != '0' && a != '1'))
		    {
		    	printf("Sorry, but I only recognize 1 and 0 (without SPACE). Please enter again: ");
		    	for(; p < 40; p++)
	            l[p] = 0;
				gets(l);
		        a = l[0];
			}
			return a;
}
