#include <limits.h>
#include "gtest/gtest.h"

#include <chrono>
#include <thread>

#include "Neon/core/core.h"


namespace chronoTest {

bool isInRagnge(double A, double B)
{
    double diff = std::abs(A - B);
    double max = std::max(A, B);

    if (diff > (max / 10.0 * 1)) {
        return false;
    }
    return true;
}


void testSecWithSleep(int sec)
{
    Neon::Timer_sec timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::seconds(sec));
    timer.stop();
    auto time = timer.time();
    ASSERT_TRUE(isInRagnge(sec, time)) << time << " vs " << sec << std::endl;
}

void testMsWithSleep(int sec)
{
    Neon::Timer_ms timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::seconds(sec));
    timer.stop();
    auto time = (timer.time()) / 1000.0;
    ASSERT_TRUE(isInRagnge(sec, time)) << time << " vs " << sec << std::endl;
}

void testUsWithSleep(int sec)
{
    Neon::Timer_us timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::seconds(sec));
    timer.stop();
    auto time = (timer.time()) / (1000.0 * 1000.0);
    ASSERT_TRUE(isInRagnge(sec, time)) << time << " vs " << sec << std::endl;
}

void testSecWithUSleep(int usec)
{
    Neon::Timer_sec timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
    timer.stop();
    auto time = timer.time() * (1000.0 * 1000.0);
    ASSERT_TRUE(isInRagnge(usec, time)) << time << " vs " << usec << std::endl;
}

void testMSecWithUSleep(int usec)
{
    Neon::Timer_ms timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
    timer.stop();
    auto time = timer.time() * (1000.0);
    ASSERT_TRUE(isInRagnge(usec, time)) << time << " vs " << usec << std::endl;
}

void testUSecWithUSleep(int usec)
{
    Neon::Timer_us timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
    timer.stop();
    auto time = timer.time();
    ASSERT_TRUE(isInRagnge(usec, time)) << time << " vs " << usec << std::endl;
}

};  // namespace chronoTest


TEST(Chrono, testSecWithSleep)
{
    chronoTest::testSecWithSleep(1);
    chronoTest::testSecWithSleep(2);
    chronoTest::testSecWithSleep(3);
}

TEST(Chrono, testMsWithSleep)
{
    chronoTest::testMsWithSleep(1);
    chronoTest::testMsWithSleep(2);
    chronoTest::testMsWithSleep(3);
}

TEST(Chrono, testUsWithSleep)
{
    chronoTest::testUsWithSleep(1);
    chronoTest::testUsWithSleep(2);
    chronoTest::testUsWithSleep(3);
}

TEST(Chrono, testSecWithUSleep)
{
    chronoTest::testSecWithUSleep(1000000);
    chronoTest::testSecWithUSleep(2000000);
    chronoTest::testSecWithUSleep(3000000);
}

TEST(Chrono, testMSecWithUSleep)
{
    chronoTest::testMSecWithUSleep(1000000);
    chronoTest::testMSecWithUSleep(2000000);
    chronoTest::testMSecWithUSleep(3000000);
}

TEST(Chrono, testUSecWithUSleep)
{
    chronoTest::testUSecWithUSleep(1000000);
    chronoTest::testUSecWithUSleep(2000000);
    chronoTest::testUSecWithUSleep(3000000);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
