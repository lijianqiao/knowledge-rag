---
title: 应用部署 FAQ
tags: [运维, FAQ, 部署]
version: 1.0
updated: 2026-05-28
---

# 应用部署 FAQ

## Q1：kubectl rollout 一直 pending 怎么办？

**A：** 常见原因：

1. 镜像拉取失败 → `kubectl describe pod` 看 Events
2. 资源不足 → `kubectl describe node` 看 Allocatable
3. PDB 阻止 → 检查 `PodDisruptionBudget`

```bash
kubectl get pods -n prod -l app=order-service
kubectl describe pod <pod-name> -n prod
```

## Q2：ConfigMap 更新后 Pod 没生效？

**A：** ConfigMap 挂载不会自动热更新（除非 subPath 未使用且应用 watch）。标准做法：

```bash
kubectl rollout restart deployment/order-service -n prod
```

## Q3：如何在预发验证后再上生产？

**A：** 流程：MR 合并 → CI 构建镜像 → 部署 staging → QA 签字 → 变更窗口部署 prod 灰度 10% → 观察 30min → 全量。

## Q4：Helm upgrade 失败如何回滚？

**A：**

```bash
helm history order-service -n prod
helm rollback order-service <revision> -n prod
```

## Q5：Pod OOMKilled 怎么排查？

**A：** 查看 `kubectl describe pod` 中 Last State，对比 limits 与实际用量；用 `kubectl top pod` 或 Prometheus 容器内存曲线。调整 requests/limits 或排查内存泄漏。
