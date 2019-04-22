#include <iostream>
#include <vector>
#include <numeric>
#include <limits>

using namespace std;

/** 请完成下面这个函数，实现题目要求的功能 **/
 /** 当然，你也可以不按照这个模板来作答，完全按照自己的想法来 ^-^  **/

long factorial(long a){
    long long b=1;//定义变量b
    for(int i=1;i<=a;i++)//计算阶乘 
       b*=i;
    return b;//返回值得到b=a！ 
}
int combinator(int n, int k)
{	
    n = n-1;
    k = k-1;
    if(n<=0){
        return 0;
    }
    if(k>n){k=n;}
	return factorial(n)/(factorial(k)*factorial(n-k));
}


int ballAllocate(int m, int n, int k) {
    int sum = 0;
    for(int i=1;i<=k-1;i++){
        int a = combinator(m,i);
        int b = combinator(n,k-i);
        cout<< i<<" "<<a<<" "<<b<< endl;
        sum = sum + combinator(m,i) + combinator(n,k-i);
    }

    return sum%10000;

}

int main() {
    int res;

    int _m=3;
    // cin >> _m;
    // cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');


    int _n=4;
    // cin >> _n;
    // cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');


    int _k=3;
    // cin >> _k;
    // cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');

    int a = combinator(4,2);
    cout<< a << endl;
    
    // res = ballAllocate(0,1,3);
    // cout << res << endl;
    
    return 0;

}