#include<iostream>
#include<conio.h>
#include <stdio.h>
#include <Windows.h>
#include <unistd.h>
#include<vector>
#include <bits/stdc++.h>
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#define DISABLE_NEWLINE_AUTO_RETURN  0x0008
using namespace std;

void addition();

void subtraction();

void multiplication();

void division();

int main()
{
sleep(.5);
DWORD l_mode;
HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
GetConsoleMode(hStdout,&l_mode);
SetConsoleMode( hStdout, l_mode |
          ENABLE_VIRTUAL_TERMINAL_PROCESSING |
          DISABLE_NEWLINE_AUTO_RETURN );

cout<<"\n\n\t\t WELCOME TO NOOB CALCULATOR\n";
cout<<"\t\t============================"<<endl<<endl<<endl;
cout<<"Before using Noob Calculator I want to ask something \n\nDo you have a calculator with you(y/n): ";
char c ;
std::cin >> c;
if(c=='y'){
  cout<<"\nThen why are you using this calculator you idiot!\n";
  getch();
  cout<<"Just kidding lets use Noob Calculator"<<endl<<endl;
  getch();
}else
{
  cout<<"\nOk you should buy one then , anyway lets use Noob Calculator!"<<endl<<endl;;
  getch();
}
cout<<"============================================================"<<endl<<endl;
do{
start:
cout<<"\nEnter the Operation You Want to do(+,-,x,/): ";
std::cin >> c;

switch (c)
{
  case '+': addition();
  break;
  case '-': subtraction();
  break;
  case 'x': multiplication();
  break;
  case '/': division();
  break;
  default:cout<<"Invalid Input!\n";
      getch();
      cout<<"Input valid operator you NOOB!"<<endl<<endl;
      cout<<"============================================================"<<endl<<endl;
      goto start;
      getch();
  break;
}
cout<<"Do you want to use the calculator again(y/n): ";
cin >> c;
}while (c=='Y'|| c=='y');
cout << "\nThank U for Using ME! Don't Forget to Buy a Calculator :)" <<endl;
return 0;
}

void division()
{
  cout<<"\n============================================================"<<endl<<endl;
  cout<<"\n\t\t DIVISION\n";
cout<<"\t============================"<<endl<<endl;
cout<<"Enter two numbers: ";
int a,b;
vector <int> vec;
vector <int> ans;
scanf("%d %d",&a,&b);
int a1=a;
if(a<b)
  cout<< a << " % " << b << " = " << 0 << endl;
else{

  while(a!= 0){
    vec.push_back(a%10);
    a/=10;
  }
  reverse(vec.begin(), vec.end());

while(vec.size()!=0){
  if(vec.size() == 1){
    if(b > vec.at(0)){
      vec.erase(vec.begin());
      break;
    }
    
  }
  while(b > vec.at(0))
  {
  vec.at(0) = vec.at(0)*10 + vec.at(1);
  vec.erase(vec.begin()+1);
  }
  cout << endl;
  for(int i = 1;i<=10000;i++)
  {
    
    if(b*i == vec.at(0)){
        sleep(1);
      cout<< b << " x " << i << " = " << vec.at(0) << endl;
      ans.push_back(i);
      sleep(1);
      cout << "Reminder " << (vec.at(0) - b*i) << " And the digit is "<< i<< endl<<endl;
      vec.erase(vec.begin());  
      break; 
    }
      
    if(b*i > vec.at(0)){
        sleep(1);
    cout<< b << " x " << i << " > " << vec.at(0) << endl;
    ans.push_back(i-1);
      sleep(1);
    cout << "Reminder " << (vec.at(0) - (b*(i-1))) << " And the digit is "<< i-1 <<endl<<endl;
    vec.at(0) = (vec.at(0) - (b*(i-1)));  
    break;
    }
      sleep(1);
    cout<< b << " x " << i << " < " << vec.at(0) << endl; 
  }
}

printf("%d / %d is ",a1,b);
for(size_t i = 0;i< ans.size() ; i++)
{
  cout << ans.at(i); 
}
printf(" and the Reminder is %d\n",a1%b);
  cout<<"\n============================================================"<<endl<<endl;
  getch();
}
}

void multiplication()
{

  cout<<"\n============================================================"<<endl<<endl;
  cout<<"\n\t\t MULTIPLICATION\n";
cout<<"\t============================"<<endl<<endl;
    int multi_table[10][10][3]=
    {
        {{1,1,1},{1,2,2},{1,3,3},{1,4,4},{1,5,5},{1,6,6},{1,7,7},{1,8,8},{1,9,9},{1,10,10}},
        {{2,1,2},{2,2,4},{2,3,6},{2,4,8},{2,5,10},{2,6,12},{2,7,14},{2,8,16},{2,9,18},{2,10,20}},
        {{3,1,3},{3,2,6},{3,3,9},{3,4,12},{3,5,15},{3,6,18},{3,7,21},{3,8,24},{3,9,27},{3,10,30}},
        {{4,1,4},{4,2,8},{4,3,12},{4,4,16},{4,5,20},{4,6,24},{4,7,28},{4,8,32},{4,9,36},{4,10,40}},
        {{5,1,5},{5,2,10},{5,3,15},{5,4,20},{5,5,25},{5,6,30},{5,7,35},{5,8,40},{5,9,45},{5,10,50}},
        {{6,1,6},{6,2,12},{6,3,18},{6,4,24},{6,5,30},{6,6,36},{6,7,42},{6,8,48},{6,9,74},{6,10,60}},
        {{7,1,7},{7,2,14},{7,3,21},{7,4,28},{7,5,35},{7,6,42},{7,7,49},{7,8,56},{7,9,63},{7,10,70}},
        {{8,1,8},{8,2,16},{8,3,24},{8,4,32},{8,5,40},{8,6,48},{8,7,56},{8,8,64},{8,9,72},{8,10,80}},
        {{9,1,9},{9,2,18},{9,3,27},{9,4,36},{9,5,45},{9,6,54},{9,7,63},{9,8,72},{9,9,81},{9,10,90}},
        {{10,1,10},{10,2,20},{10,3,30},{10,4,40},{10,5,50},{10,6,60},{10,7,70},{10,8,80},{10,9,90},{10,10,100}}
    };
    start:
    cout<<"Enter the numbers in range{10,10}(I am an idiot i dont know many multipication): ";
    int a,b;
    cin>>a >> b;
    if(a> 10 || b>10){
        cout<<"I told you I am an Idiot! Give me small numbers" << endl;
        getch();
        goto start;
    }
    printf("\nShowing the table of %d\n\n",a);
    for(int i = 0;i<10; i++)
    {
        sleep(1);
        cout<<multi_table[a-1][i][0] << " x " << multi_table[a-1][i][1] << " = " << multi_table[a-1][i][2] << endl<<endl;
    }
    cout << "Getting the multiplication in Row " << b << "...."<< endl<<endl;
    sleep(1);
    cout << "The answer is ";
    sleep(1);
    cout << a << " x " << b <<" = "  <<   multi_table[a-1][b-1][2] << endl;
    getch();
    cout<< "\nsry for taking time I was just looking the multiplication table"<<endl<<endl;
    cout<<"\n============================================================"<<endl<<endl;
    getch();
}

void subtraction()
{
     cout<<"\n============================================================"<<endl<<endl;
    cout<<"\n\t\t SUBTRACTION\n";
  cout<<"\t============================"<<endl<<endl;
    cout<<"Enter Two numbers: ";
    int a,b,temp=0,a1,b1,temp2;
    bool j =0;
    scanf("%d %d", &a, &b);
    cout << endl;
    a1=a;
    b1=b;
    if(a<b){
      temp2 = a;
      a = b;
      b = temp2;
      j = 1;
    }
    int n = a/b;
    
     for(int i = 1;i<=n;i++)
     {
         sleep(1);
         cout<<b<<"+";
     }
    cout << a%b << "=" << a <<  endl;
    printf("Here we have %d %d's in %d\n\nTherefore we remove one %d from %d we get\n", n,b,a,b,a);
    for (int i = 0; i < n-1; i++)
    {
        sleep(1);
        cout<<b<<"+";
        temp +=b;
    }
    cout << a%b << "=" << temp + a%b  <<  endl;
    if(j==1)
    printf("Because the %d is smaller than %d so a negative sign will present in the Answer\n",a1,b1);
    cout << "\nThere for the answer for " << a1 << " - " << b1 <<" is ";
    (j == 1)?cout<<"-":cout <<" ";
    cout <<  temp + a%b << endl;
    cout<<"\n============================================================"<<endl<<endl; 
    getch();
}

void addition()
{
    cout<<"\n============================================================"<<endl<<endl;
    cout<<"\n\t\t ADDITION\n";
  cout<<"\t============================"<<endl<<endl;
    cout<<"Enter two numbers: ";
    int a,b;
    scanf("%d %d",&a,&b);
    int temp = a;
    for(int i =0; i <=b;i++)
    { 
        cout<< temp;
        if(temp <(a+b)){
        sleep(1);
        cout<<'+';
        sleep(1);
        cout << 1;
        sleep(1); 
        temp++;
        std::cout << "\x1B[2K\r";
        }
    }
    sleep(1);
    cout<<endl;
    cout<< "\nYeah! we got the Answer it is " << temp << "\nIt was tough to add" <<endl;
    cout<<"\n============================================================"<<endl<<endl;
    getch();

}