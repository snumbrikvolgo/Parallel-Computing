#include <iostream>
long long fact(int x)
{
    long long p = 1;
    for (int i = x; i > 0; i--)
    {
        p *= i;
    }
    return p;
}

int main()
{
    double e = 2;
    double sum = 0;

    for (int i = 2; i < 30; i++)
    {

        sum = static_cast<double> (1.0/fact(i));
        e += sum;
    }
    std::cout << e << "\n";
}
